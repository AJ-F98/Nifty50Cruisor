# src/predict.py
import joblib
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from src.features import engineer_features

def predict_tomorrow():
    with open("models/features.json") as f:
        features = json.load(f)
    scaler = joblib.load("models/scaler.pkl")
    model = tf.keras.models.load_model("models/nifty50_lstm.h5")

    # Last 90 days to ensure enough data for 30 sequence + indicators
    df = yf.download("^NSEI", period="90d", progress=False)
    vix = yf.download("^INDIAVIX", period="90d", progress=False)
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    if 'Close' in vix.columns:
        df['VIX_Close'] = vix['Close']
    else:
        df['VIX_Close'] = 0

    df = df.reset_index()
    df = engineer_features(df).dropna()
    
    print("Predict DF shape:", df.shape)

    if df.empty:
        print("Error: DF is empty after engineering features")
        return "Error", 0.0, [0, 0, 0]
        
    # Ensure features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        return "Error", 0.0, [0, 0, 0]

    seq = scaler.transform(df[features].tail(30))
    seq = seq.reshape(1, 30, -1)

    prob = model.predict(seq, verbose=0)[0]
    pred_class = np.argmax(prob)
    confidence = prob[pred_class]

    mapping = {0: "Sideways", 1: "Bullish", 2: "Bearish"}
    return mapping[pred_class], confidence, prob.tolist()

def predict_next_close():
    df = None  # Define upfront so it's always in scope
    try:
        # Load model stuff
        with open("models/return_features.json") as f:
            features = json.load(f)
        scaler = joblib.load("models/return_scaler.pkl")
        target_scaler = joblib.load("models/return_target_scaler.pkl")
        model = tf.keras.models.load_model("models/nifty50_return_lstm.h5")

        # Download data
        df = yf.download("^NSEI", period="100d", progress=False, auto_adjust=True)
        vix = yf.download("^INDIAVIX", period="100d", progress=False)

        # Flatten columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        df['VIX_Close'] = vix['Close'].reindex(df.index).fillna(method='ffill').fillna(15.0)
        df = df.reset_index()
        df = engineer_features(df).dropna()

        if len(df) < 35:
            raise ValueError("Not enough clean data")

        # Final sequence
        seq = scaler.transform(df[features].tail(30))
        seq = seq.reshape(1, 30, -1)

        pred_scaled = model.predict(seq, verbose=0)
        pred_return = target_scaler.inverse_transform(pred_scaled)[0][0]

        today_close = float(df['Close'].iloc[-1])
        predicted_close = today_close * (1 + pred_return)

        print(f"SUCCESS → Today: ₹{today_close:,.2f} | Move: {pred_return*100:+.3f}% | Tomorrow: ₹{predicted_close:,.2f}")
        return round(predicted_close, 2)

    except Exception as e:
        print(f"PREDICTION FAILED → Error: {e}")
        # FALLBACK 1: Use last known close
        try:
            last = yf.download("^NSEI", period="2d", progress=False)['Close'].iloc[-1]
            print(f"Falling back to latest close: ₹{last:,.2f}")
            return round(float(last), 2)
        except:
            print("Total failure — returning 25200 as last resort")
            return 25200.0
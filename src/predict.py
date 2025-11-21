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
    # print("Predict DF columns:", df.columns)

    if df.empty:
        print("Error: DF is empty after engineering features")
        return "Error", 0.0, [0, 0, 0]
        
    # Ensure features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        return "Error", 0.0, [0, 0, 0]

    seq = scaler.transform(df[features].tail(30))
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
    # print("Predict DF columns:", df.columns)

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

def predict_price():
    with open("models/price_features.json") as f:
        features = json.load(f)
    scaler = joblib.load("models/price_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")
    model = tf.keras.models.load_model("models/nifty50_price_lstm.h5")

    # Last 90 days
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

    if df.empty:
        return 0.0

    seq = scaler.transform(df[features].tail(30))
    seq = seq.reshape(1, 30, -1)

    pred_scaled = model.predict(seq, verbose=0)
    pred_price = target_scaler.inverse_transform(pred_scaled)[0][0]
    
    return float(pred_price)
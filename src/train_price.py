# src/train_price.py

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now imports work perfectly
import joblib
import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from src.data_loader import load_nifty_data
from src.features import engineer_features
from src.model import build_price_model

SEQUENCE_LENGTH = 30

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


if __name__ == "__main__":
    print("Starting Nifty50 Return-Based Price Model Training...")
    
    # Load and engineer features
    df = load_nifty_data()
    df = engineer_features(df)
    
    # TARGET: Next-day percentage return (this is the PRO way)
    df['next_return'] = df['Close'].pct_change().shift(-1)
    df = df.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close',
                'return', 'return_5', 'return_10', 'return_20',
                'vol_5', 'vol_10', 'vol_20', 'rsi', 'macd', 'macd_signal',
                'bb_width', 'close_to_high', 'open_to_close',
                'gap_up', 'gap_down', 'doji', 'dow', 'month', 'is_month_end']

    print(f"Training on {len(df)} days of data...")
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Scale target returns
    return_scaler = RobustScaler()
    y_scaled = return_scaler.fit_transform(df[['next_return']])

    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH)

    split = int(0.85 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"Training sequences: {len(X_train)}, Test: {len(X_test)}")

    # Build and train model
    model = build_price_model((SEQUENCE_LENGTH, X_seq.shape[2]))
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)],
        verbose=1
    )

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # Save everything
    model.save("models/nifty50_return_lstm.h5")
    joblib.dump(scaler, "models/return_scaler.pkl")
    joblib.dump(return_scaler, "models/return_target_scaler.pkl")
    with open("models/return_features.json", "w") as f:
        json.dump(features, f)

    # Final evaluation
    pred_scaled = model.predict(X_test, verbose=0)
    pred_return = return_scaler.inverse_transform(pred_scaled)
    actual_return = return_scaler.inverse_transform(y_test)
    mae_pct = np.mean(np.abs(pred_return - actual_return)) * 100

    print(f"Test MAE: {mae_pct:.3f}% (daily return prediction error)")

# src/train_price.py
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
    # Load & engineer
    df = load_nifty_data()
    df = engineer_features(df)
    
    # Target: Next day's Close price
    df['target_price'] = df['Close'].shift(-1)
    df = df.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close',
                'return', 'return_5', 'return_10', 'return_20',
                'vol_5', 'vol_10', 'vol_20', 'rsi', 'macd', 'macd_signal',
                'bb_width', 'close_to_high', 'open_to_close',
                'gap_up', 'gap_down', 'doji', 'dow', 'month', 'is_month_end']

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Scale target (optional but recommended for LSTM convergence)
    target_scaler = RobustScaler()
    y_scaled = target_scaler.fit_transform(df[['target_price']])

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH)

    split = int(0.85 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = build_price_model((SEQUENCE_LENGTH, X_seq.shape[2]))
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=100,
              batch_size=32,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=1)

    # Save
    model.save("models/nifty50_price_lstm.h5")
    joblib.dump(scaler, "models/price_scaler.pkl")
    joblib.dump(target_scaler, "models/target_scaler.pkl")
    with open("models/price_features.json", "w") as f:
        json.dump(features, f)

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTraining Complete! Test MAE: {test_mae:.4f}")
    print("Model saved to models/")

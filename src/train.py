# src/train.py
import joblib
import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.utils import to_categorical
from src.data_loader import load_nifty_data
from src.features import engineer_features
from src.model import build_model

SEQUENCE_LENGTH = 30
THRESHOLD = 0.0075

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
    
    # Target
    df['future_return'] = df['Close'].pct_change().shift(-1)
    df['target'] = np.where(df['future_return'] > THRESHOLD, 1,
                   np.where(df['future_return'] < -THRESHOLD, -1, 0))
    df = df.dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close',
                'return', 'return_5', 'return_10', 'return_20',
                'vol_5', 'vol_10', 'vol_20', 'rsi', 'macd', 'macd_signal',
                'bb_width', 'close_to_high', 'open_to_close',
                'gap_up', 'gap_down', 'doji', 'dow', 'month', 'is_month_end']

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])

    X_seq, y_seq = create_sequences(X_scaled, df['target'].values, SEQUENCE_LENGTH)
    y_cat = to_categorical(y_seq, num_classes=3)

    split = int(0.85 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_cat[:split], y_cat[split:]

    model = build_model((SEQUENCE_LENGTH, X_seq.shape[2]))
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=100,
              batch_size=32,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=1)

    # Save
    model.save("models/nifty50_lstm.h5")
    joblib.dump(scaler, "models/scaler.pkl")
    with open("models/features.json", "w") as f:
        json.dump(features, f)

    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"\nTraining Complete! Test Accuracy: {test_acc:.4f}")
    print("Model saved to models/")
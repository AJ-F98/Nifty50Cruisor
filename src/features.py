# src/features.py
import pandas as pd
import numpy as np
import pandas_ta as ta

def engineer_features(df):
    df = df.copy()
    # Ensure columns are numeric
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df['return'] = df['Close'].pct_change()
    for period in [5, 10, 20]:
        df[f'return_{period}'] = df['Close'].pct_change(period)
        df[f'vol_{period}'] = df['return'].rolling(period).std()

    df['rsi'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    if macd is not None:
        # MACD columns might vary
        # print("MACD columns:", macd.columns)
        if 'MACD_12_26_9' in macd.columns:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
        else:
            # Fallback or take first/second columns
            df['macd'] = macd.iloc[:, 0]
            df['macd_signal'] = macd.iloc[:, 2] # Signal usually 3rd? No, MACD, Histogram, Signal usually.
            # ta.macd returns: MACD, Histogram, Signal. 
            # Let's check documentation or assume standard names.
            pass

    bb = ta.bbands(df['Close'], length=20)
    if bb is not None:
        # print("BB columns:", bb.columns)
        if 'BBB_20_2.0' in bb.columns:
            df['bb_width'] = bb['BBB_20_2.0']
        elif 'BBU_20_2.0' in bb.columns and 'BBL_20_2.0' in bb.columns and 'BBM_20_2.0' in bb.columns:
            df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
        else:
            # Fallback: try to find columns ending with specific suffixes
            lower = bb.filter(like='BBL').iloc[:, 0]
            upper = bb.filter(like='BBU').iloc[:, 0]
            mid = bb.filter(like='BBM').iloc[:, 0]
            df['bb_width'] = (upper - lower) / mid

    df['close_to_high'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-6)
    df['open_to_close'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-6)
    df['gap_up'] = (df['Open'] > df['Close'].shift(1) * 1.005).astype(int)
    df['gap_down'] = (df['Open'] < df['Close'].shift(1) * 0.995).astype(int)
    df['doji'] = (abs(df['Open'] - df['Close']) / (df['High'] - df['Low'] + 1e-6) < 0.1).astype(int)

    if 'Date' in df.columns:
        df['dow'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['month'] = pd.to_datetime(df['Date']).dt.month
        df['is_month_end'] = pd.to_datetime(df['Date']).dt.is_month_end.astype(int)
    elif 'Date' in df.index.names:
        df['dow'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_month_end'] = df.index.is_month_end.astype(int)
    else:
        # Try to find date column
        pass

    return df
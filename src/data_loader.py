# src/data_loader.py
import yfinance as yf
import pandas as pd
from datetime import datetime

def load_nifty_data(start_date="2007-01-01"):
    end_date = datetime.today().strftime('%Y-%m-%d')
    print("Downloading Nifty50 (^NSEI)...")
    nifty = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
    print("Downloading India VIX (^INDIAVIX)...")
    vix = yf.download("^INDIAVIX", start=start_date, end=end_date, progress=False)
    
    # Handle MultiIndex columns if present
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        
    print("Nifty columns:", nifty.columns)
    print("Nifty head:", nifty.head())
    
    df = nifty.copy()
    # Ensure Close column exists
    if 'Close' not in df.columns:
        # Fallback or error
        print("Error: 'Close' column not found in Nifty data")
        print(df.columns)
        return pd.DataFrame()

    if 'Close' in vix.columns:
        df['VIX_Close'] = vix['Close']
    else:
        print("Warning: 'Close' column not found in VIX data")
        df['VIX_Close'] = 0 # Or handle appropriately

    df = df.dropna(subset=['Close'])
    return df.reset_index()
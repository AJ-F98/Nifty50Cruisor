# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from src.predict import predict_tomorrow, predict_price
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="Nifty50 Predictor", layout="centered")

st.title("NIFTY50 Next-Day Direction Predictor")
st.markdown("##### Bi-LSTM + India VIX + 25 Smart Features | Live Daily Update")

# 1. Live Price
try:
    nifty_live = yf.download("^NSEI", period="1d", interval="1m", progress=False)
    if isinstance(nifty_live.columns, pd.MultiIndex):
        nifty_live.columns = nifty_live.columns.get_level_values(0)
    
    if not nifty_live.empty:
        current_price = nifty_live['Close'].iloc[-1]
        st.metric("Nifty50 Live Price", f"₹{current_price:,.2f}")
    else:
        st.warning("Could not fetch live price.")
except Exception as e:
    st.error(f"Error fetching live price: {e}")

# 2. Predictions
direction, confidence, probs = predict_tomorrow()
predicted_price = predict_price()

color = {"Bullish": "🟢", "Bearish": "🔴", "Sideways": "⚪"}[direction]
st.markdown(f"### {color} **Tomorrow: {direction.upper()}**")
st.progress(confidence)
st.write(f"**Confidence: {confidence:.1%}**")

st.markdown(f"### **Predicted Close: ₹{predicted_price:,.2f}**")

col1, col2, col3 = st.columns(3)
col1.metric("Bullish ↑", f"{probs[1]:.1%}")
col2.metric("Bearish ↓", f"{probs[2]:.1%}")
col3.metric("Sideways →", f"{probs[0]:.1%}")

# 3. Save Prediction
prediction_file = "predictions.txt"
today_str = datetime.now().strftime('%Y-%m-%d')
log_entry = f"{today_str},{predicted_price:.2f}\n"

# Check if today's prediction is already saved to avoid duplicates
if os.path.exists(prediction_file):
    with open(prediction_file, "r") as f:
        lines = f.readlines()
        if not any(today_str in line for line in lines):
            with open(prediction_file, "a") as f:
                f.write(log_entry)
else:
    with open(prediction_file, "w") as f:
        f.write("Date,Predicted_Close\n")
        f.write(log_entry)

# 4. History Table (Last 5 Days)
st.subheader("Last 5 Days Performance")
try:
    # Get actual history
    history = yf.download("^NSEI", period="10d", progress=False)
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    
    history = history.reset_index().sort_values('Date', ascending=False).head(5)
    history['Date'] = history['Date'].dt.strftime('%Y-%m-%d')
    
    # Get predicted history
    preds = {}
    if os.path.exists(prediction_file):
        with open(prediction_file, "r") as f:
            for line in f.readlines()[1:]: # Skip header
                parts = line.strip().split(',')
                if len(parts) == 2:
                    preds[parts[0]] = float(parts[1])
    
    # Combine
    data = []
    for _, row in history.iterrows():
        date = row['Date']
        actual = row['Close']
        predicted = preds.get(date, None)
        diff = actual - predicted if predicted else None
        data.append({
            "Date": date,
            "Actual Close": f"₹{actual:,.2f}",
            "Predicted Close": f"₹{predicted:,.2f}" if predicted else "N/A",
            "Difference": f"₹{diff:,.2f}" if diff else "N/A"
        })
        
    st.table(pd.DataFrame(data))

except Exception as e:
    st.error(f"Error loading history: {e}")

# Chart
chart_data = yf.download("^NSEI", period="3mo", progress=False)['Close']
st.line_chart(chart_data)

st.caption(f"Last updated: {datetime.now().strftime('%b %d, %Y %I:%M %p')}")
st.info("Model Accuracy: ~63%+ on unseen data | Threshold: ±0.75%")
# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from src.predict import predict_tomorrow, predict_next_close
from datetime import datetime
import os
import logging
logging.basicConfig(level=logging.INFO)

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
predicted_price = predict_next_close()

# Convert numpy → Python
confidence = float(confidence)
probs = [float(p) for p in probs]

color = {"Bullish": "green", "Bearish": "red", "Sideways": "gray"}[direction]
st.markdown(f"### <span style='color:{color}'>**Tomorrow: {direction.upper()}**</span>", unsafe_allow_html=True)
st.progress(confidence)
st.write(f"**Confidence: {confidence:.1%}**")

st.markdown(f"### **Predicted Tomorrow's Close: ₹{predicted_price:,.2f}**")

col1, col2, col3 = st.columns(3)
col1.metric("Bullish", f"{probs[1]:.1%}")
col2.metric("Bearish", f"{probs[2]:.1%}")
col3.metric("Sideways", f"{probs[0]:.1%}")

# 3. Save Prediction
prediction_file = "predictions.txt"
today_str = datetime.now().strftime('%Y-%m-%d')
log_entry = f"{today_str},{predicted_price:.2f}\n"

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

# 4. CORRECTED: Next-Day Prediction Accuracy Table (t → t+1)
st.subheader("Next-Day Prediction Accuracy")

try:
    # Load all predictions
    preds = {}
    if os.path.exists(prediction_file):
        with open(prediction_file, "r") as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    preds[parts[0]] = float(parts[1])

    # Get recent actual closes
    history = yf.download("^NSEI", period="20d", progress=False)
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    closes = history['Close'].dropna()
    closes.index = pd.to_datetime(closes.index)

    data = []
    errors = []

    for i in range(len(closes) - 1):
        pred_date = closes.index[i].strftime('%Y-%m-%d')
        actual_date = closes.index[i+1].strftime('%Y-%m-%d')
        actual_close = closes.iloc[i+1]
        predicted = preds.get(pred_date)

        if predicted is not None:
            diff = actual_close - predicted
            errors.append(abs(diff))
            color = "🟢" if abs(diff) <= 50 else "🟡" if abs(diff) <= 100 else "🔴"
            data.append({
                "Prediction Made": pred_date,
                "For Day": actual_date,
                "Predicted": f"₹{predicted:,.0f}",
                "Actual": f"₹{actual_close:,.0f}",
                "Error": f"{color} {diff:+,.0f}"
            })

    if data:
        df = pd.DataFrame(data).head(10)
        st.table(df)

        mae = sum(errors) / len(errors) if errors else 0
        win_rate = sum(1 for e in errors if e <= 75) / len(errors) * 100 if errors else 0

        col1, col2 = st.columns(2)
        col1.success(f"**Average Error (MAE): ₹{mae:,.0f}**")
        col2.success(f"**Win Rate (±0.75%): {win_rate:.1f}%**")
    else:
        st.info("Waiting for first next-day result...")

except Exception as e:
    st.error(f"Error loading accuracy table: {e}")

# Chart
st.subheader("Nifty50 - 3 Month Trend")
chart_data = yf.download("^NSEI", period="3mo", progress=False)['Close']
st.line_chart(chart_data)

st.caption(f"Last updated: {datetime.now().strftime('%b %d, %Y • %I:%M %p')}")
st.info("Model: Bi-LSTM Return Predictor + Direction Classifier | Threshold: ±0.75% | Live since Nov 2025")
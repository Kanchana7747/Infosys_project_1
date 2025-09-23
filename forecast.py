import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Streamlit page config
st.set_page_config(page_title="Air Quality AQI Forecast", layout="wide")
st.title("🌫️ Air Quality AQI Forecast")

# State selection only
stations = ["Maharashtra", "Delhi", "Rajasthan", "Gujarat", "Tamil Nadu", "Karnataka", "West Bengal", "Uttar Pradesh"]
station = st.sidebar.selectbox("Select Station:", stations)

pollutant = "AQI"  # Fixed pollutant
forecast_days = 7

# Load and filter data for selected state
df = pd.read_csv(r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\city_hour.csv", parse_dates=["date"])
df = df[df['State'] == station].sort_values("date")

# Color coding function for AQI levels
def get_aqi_color(aqi):
    if aqi <= 50:
        return '#d5f6e3'  # Good
    elif aqi <= 100:
        return '#f9f7d8'  # Moderate
    elif aqi <= 150:
        return '#fff3cd'  # Unhealthy for Sensitive
    else:
        return '#fbe7c6'  # Unhealthy

# Check data availability and perform forecasting
if df.empty:
    st.error(f"No data available for {station}")
else:
    values = df[pollutant].dropna().values.reshape(-1, 1)
    n_rows = len(values)

    if n_rows < 2:
        st.warning(f"Not enough data to forecast {pollutant}. Need at least 2 rows.")
    else:
        window_size = min(30, n_rows - 1)

        X, y = [], []
        for i in range(n_rows - window_size):
            X.append(values[i:i + window_size])
            y.append(values[i + window_size])

        X = np.array(X).reshape(-1, window_size, 1)
        y = np.array(y)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        y_scaled = scaler.transform(y.reshape(-1, 1))

        model = Sequential([
            LSTM(64, activation='relu', input_shape=(window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_scaled, y_scaled, epochs=50, batch_size=16, verbose=0)

        forecast_scaled = []
        last_seq = X_scaled[-1].reshape(1, window_size, 1)
        for _ in range(forecast_days):
            pred = model.predict(last_seq, verbose=0)
            forecast_scaled.append(pred[0, 0])
            pred_seq = pred.reshape(1, 1, 1)
            last_seq = np.append(last_seq[:, 1:, :], pred_seq, axis=1)

        forecast_values = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        forecast_dates = [(datetime.today() + timedelta(days=i)).strftime('%a') for i in range(1, forecast_days + 1)]

        st.subheader(f"📅 7-Day {pollutant} Forecast for {station}")

        # Display forecast with color-coded boxes horizontally
        cols = st.columns(forecast_days)
        for i, col in enumerate(cols):
            color = get_aqi_color(forecast_values[i])
            col.markdown(
                f"<div style='background:{color}; padding:20px; border-radius:12px; text-align:center;'>"
                f"<span style='font-weight:bold; font-size:18px;'>{forecast_dates[i]}</span><br>"
                f"AQI {forecast_values[i]:.2f}"
                f"</div>", unsafe_allow_html=True
            )

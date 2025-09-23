import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv(r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\city_hour.csv", parse_dates=["date"])
df = df.sort_values("date")

# Pick pollutant columns
pollutants = ["AQI", "NO2", "O3", "CO", "PM2.5", "PM10"]

window_size = 30
forecast_dict = {}

# -----------------------------
# 2. Train & Forecast per pollutant
# -----------------------------
for pollutant in pollutants:
    print(f"\n🔮 Forecasting {pollutant}...")

    values = df[pollutant].dropna().values.reshape(-1,1)  # drop NaNs
    
    # Scale values
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled_values) - window_size):
        X.append(scaled_values[i:i+window_size])
        y.append(scaled_values[i+window_size])
    X, y = np.array(X), np.array(y)

    # Build & Train LSTM
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, batch_size=16, verbose=0)

    # Forecast next 7 days
    forecast_scaled = []
    input_seq = scaled_values[-window_size:].reshape(1, window_size, 1)

    for _ in range(7):
        pred = model.predict(input_seq, verbose=0)
        forecast_scaled.append(pred[0,0])
        pred = pred.reshape(1,1,1)
        input_seq = np.append(input_seq[:,1:,:], pred, axis=1)

    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1)).flatten()
    forecast_dict[pollutant] = forecast

# -----------------------------
# 3. Print results
# -----------------------------
print("\n📅 7-Day LSTM Forecasts:")
for pollutant, values in forecast_dict.items():
    print(f"\n{pollutant}:")
    for i, val in enumerate(values, 1):
        print(f"  Day {i}: {val:.2f}")

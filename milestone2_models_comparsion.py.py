# 🛠 Time Series Forecasting Comparison: ARIMA vs Prophet vs LSTM
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# ------------------------
# 1. Load and Prepare Data
# ------------------------
df = pd.read_csv(r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\city_day.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
series = df["PM2.5"].fillna(df["PM2.5"].median())  # fill missing values

# ------------------------
# 2. Train-Test Split
# ------------------------
train_size = int(len(series) * 0.8)
train, test = series.iloc[:train_size], series.iloc[train_size:]

# ------------------------
# 3. ARIMA Model
# ------------------------
arima_model = ARIMA(train, order=(2,1,2))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test))

arima_mae = mean_absolute_error(test, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))

# ------------------------
# 4. Prophet Model
# ------------------------
df_prophet = df.reset_index()[["Date", "PM2.5"]]
df_prophet.columns = ["ds", "y"]

prophet_model = Prophet()
prophet_model.fit(df_prophet.iloc[:train_size])

future = prophet_model.make_future_dataframe(periods=len(test))
forecast_prophet = prophet_model.predict(future)

prophet_forecast = forecast_prophet.set_index("ds")["yhat"].iloc[-len(test):]

prophet_mae = mean_absolute_error(test, prophet_forecast)
prophet_rmse = np.sqrt(mean_squared_error(test, prophet_forecast))

# ------------------------
# 5. LSTM Model
# ------------------------
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))

window_size = 10
train_gen = TimeseriesGenerator(scaled_series[:train_size], scaled_series[:train_size],
                                length=window_size, batch_size=32)
test_gen = TimeseriesGenerator(scaled_series[train_size - window_size:], 
                               scaled_series[train_size - window_size:],
                               length=window_size, batch_size=32)

lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(train_gen, epochs=10, verbose=0)

lstm_preds_scaled = lstm_model.predict(test_gen)
lstm_forecast = scaler.inverse_transform(lstm_preds_scaled).flatten()

y_test_lstm = series.iloc[train_size:train_size + len(lstm_forecast)]
y_test_lstm = y_test_lstm.fillna(method="ffill")  # just in case
lstm_forecast = pd.Series(lstm_forecast).fillna(method="ffill")

lstm_mae = mean_absolute_error(y_test_lstm, lstm_forecast)
lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_forecast))

# ------------------------
# 6. Compare Models
# ------------------------
print("📊 Model Performance Comparison")
print(f"ARIMA   -> MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}")
print(f"Prophet -> MAE: {prophet_mae:.2f}, RMSE: {prophet_rmse:.2f}")
print(f"LSTM    -> MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}")

# ------------------------
# 7. Save Best Model
# ------------------------
rmse_scores = {
    "ARIMA": arima_rmse,
    "Prophet": prophet_rmse,
    "LSTM": lstm_rmse
}

best_model_name = min(rmse_scores, key=rmse_scores.get)

if best_model_name == "ARIMA":
    joblib.dump(arima_fit, "best_model.pkl")
elif best_model_name == "Prophet":
    joblib.dump(prophet_model, "best_model.pkl")
else:
    lstm_model.save("best_model.h5")

print(f"✅ Best model saved: {best_model_name}")

# 🛠 Time Series Forecasting Comparison: ARIMA vs Prophet vs Naive
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import joblib

# ------------------------
# 1. Load and Prepare Data
# ------------------------
df = pd.read_csv(r"C:\Users\slaxm\OneDrive\Documents\city_hour.csv", parse_dates=["date"])
df.set_index("date", inplace=True)
series = df["PM2.5"]

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

# Evaluate ARIMA
arima_mae = mean_absolute_error(test, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))

# ------------------------
# 4. Prophet Model
# ------------------------
df_prophet = df.reset_index()[["date", "PM2.5"]]
df_prophet.columns = ["ds", "y"]

prophet_model = Prophet()
prophet_model.fit(df_prophet.iloc[:train_size])

future = prophet_model.make_future_dataframe(periods=len(test))
forecast_prophet = prophet_model.predict(future)

prophet_rmse = np.sqrt(mean_squared_error(
    test, forecast_prophet["yhat"].iloc[train_size:]
))
prophet_mae = mean_absolute_error(test, forecast_prophet["yhat"].iloc[train_size:])

# ------------------------
# 5. Naive Forecast (Last observed value)
# ------------------------
naive_forecast = np.repeat(train.iloc[-1], len(test))
naive_rmse = np.sqrt(mean_squared_error(test, naive_forecast))
naive_mae = mean_absolute_error(test, naive_forecast)

# ------------------------
# 6. Compare Models
# ------------------------
print("📊 Model Performance Comparison")
print(f"ARIMA     -> MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}")
print(f"Prophet   -> MAE: {prophet_mae:.2f}, RMSE: {prophet_rmse:.2f}")
print(f"Naive     -> MAE: {naive_mae:.2f}, RMSE: {naive_rmse:.2f}")

# ------------------------
# 7. Save Best Model
# ------------------------
best_model_name = None
if arima_rmse <= prophet_rmse and arima_rmse <= naive_rmse:
    joblib.dump(arima_fit, "best_model.pkl")
    best_model_name = "ARIMA"
elif prophet_rmse <= arima_rmse and prophet_rmse <= naive_rmse:
    joblib.dump(prophet_model, "best_model.pkl")
    best_model_name = "Prophet"
else:
    joblib.dump(naive_forecast, "best_model.pkl")  # naive is just an array
    best_model_name = "Naive"

print(f"✅ Best model saved: {best_model_name}")

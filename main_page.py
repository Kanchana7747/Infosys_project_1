import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import sqlite3

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'> Air Quality Dashboard</h1>", unsafe_allow_html=True)

# ----------------------------
# SQLite setup for forecasts (viewable in VS Code)
# ----------------------------
DB_PATH_VIEWABLE = r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\aqi_forecasts_viewable.db"
conn_viewable = sqlite3.connect(DB_PATH_VIEWABLE, check_same_thread=False)
c_viewable = conn_viewable.cursor()

c_viewable.execute("""
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station TEXT,
    forecast_label TEXT,
    forecast_date TEXT,
    forecast_aqi REAL
)
""")
conn_viewable.commit()

# Functions
def load_default_data():
    return pd.read_csv(r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\city_day.csv")

# File Upload
uploaded_file = st.file_uploader(
    "Upload your city_hour file (optional, default dataset will be used otherwise)",
    type=["csv", "xlsx", "xls"]
)

df = load_default_data()
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif uploaded_file.name.endswith(".xls"):
            df = pd.read_excel(uploaded_file, engine="xlrd")
        st.success(f"✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Could not read uploaded file: {e}")
        st.stop()

# Clean Columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

# Standardize date column
if "Datetime" in df.columns:
    df.rename(columns={"Datetime": "date"}, inplace=True)
elif "Date" in df.columns:
    df.rename(columns={"Date": "date"}, inplace=True)
elif "date" not in df.columns:
    st.error("No 'Date' or 'Datetime' column found in dataset.")
    st.stop()
df["date"] = pd.to_datetime(df["date"], errors="coerce")

st.subheader(f"📊 Full Dataset")
st.dataframe(df.head())

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("⚙️ Controls")
location_col = next((c for c in ["State", "station_id", "City", "StationId"] if c in df.columns), None)
if not location_col:
    st.error("Dataset must have a location column: 'State', 'station_id', or 'City'.")
    st.stop()

stations = df[location_col].dropna().unique()
time_ranges = {"Last 7 Days": 7, "Last 30 Days": 30}
pollutants = ["PM2.5", "NO2", "O3", "CO", "PM10","SO2"]
forecast_options = {"24 Hours": "hourly", "7 Days": "daily", "15 Days": "daily"}

station = st.sidebar.selectbox(f"Select {location_col}", stations)
time_range = st.sidebar.selectbox("Time Range", list(time_ranges.keys()))
selected_pollutants = st.sidebar.multiselect("Select pollutants to display", pollutants)
forecast_label = st.sidebar.selectbox("Forecast Horizon", list(forecast_options.keys()))
forecast_days = {"24 Hours":24, "7 Days":7, "15 Days":15}[forecast_label]

# Filter Data
end_date = df["date"].max()
start_date = end_date - pd.Timedelta(days=time_ranges[time_range])
df_station = df[df[location_col] == station].sort_values("date")


# Data Cleaning
for col in pollutants + ["AQI"]:
    if col in df_station.columns:
        df_station[col].fillna(df_station[col].median(), inplace=True)
        Q1 = df_station[col].quantile(0.25)
        Q3 = df_station[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_station[col] = df_station[col].clip(lower=lower_bound, upper=upper_bound)

# AQI Calculation
common_breakpoints = [
    {"low":0,"high":50,"index_low":0,"index_high":50},
    {"low":51,"high":100,"index_low":51,"index_high":100},
    {"low":101,"high":200,"index_low":101,"index_high":200},
    {"low":201,"high":300,"index_low":201,"index_high":300},
    {"low":301,"high":400,"index_low":301,"index_high":400},
    {"low":401,"high":500,"index_low":401,"index_high":500}
]

breakpoints_dict = {pol: common_breakpoints for pol in pollutants}

def calculate_aqi_subindex(val, common_breakpoints):
    for bp in common_breakpoints:
        if bp["low"] <= val <= bp["high"]:
            return ((bp["index_high"] - bp["index_low"]) / (bp["high"] - bp["low"])) * (val - bp["low"]) + bp["index_low"]
    return 0

if "AQI" in df_station.columns:
    df_station["AQI_Selected"] = df_station["AQI"]
else:
    aqi_subindexes = []
    for pol in pollutants:
        if pol in df_station.columns:
            sub_index = df_station[pol].apply(lambda x: calculate_aqi_subindex(x, breakpoints_dict[pol]))
            aqi_subindexes.append(sub_index)
    df_station["AQI_Selected"] = pd.concat(aqi_subindexes, axis=1).max(axis=1) if aqi_subindexes else 0

current_aqi = df_station["AQI_Selected"].iloc[-1] if not df_station.empty else 0

# ----------------------------
# AQI Gauge
# ----------------------------
if current_aqi <= 50: aqi_category, color = "Good", "green"
elif current_aqi <= 100: aqi_category, color = "Moderate", "yellow"
elif current_aqi <= 200: aqi_category, color = "Unhealthy", "orange"
elif current_aqi <= 300: aqi_category, color = "Very Unhealthy", "red"
else: aqi_category, color = "Hazardous", "maroon"

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=current_aqi,
    title={"text": f"{aqi_category}"},
    gauge={
        "axis": {"range": [0, 500]},
        "bar": {"color": color},
        "steps": [
            {"range": [0, 50], "color": "lightgreen"},
            {"range": [51, 100], "color": "yellow"},
            {"range": [101, 200], "color": "orange"},
            {"range": [201, 300], "color": "red"},
            {"range": [301, 400], "color": "purple"},
            {"range": [401, 500], "color": "maroon"},
        ],
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)


# Pollutant Trend
df_trend = df[df[location_col] == station].copy()
latest_data_date = df_trend["date"].max()
shift_days = (datetime.now().date() - latest_data_date.date()).days
df_trend["date_aligned"] = df_trend["date"] + pd.to_timedelta(shift_days, unit="D")
df_trend = df_trend[df_trend["date_aligned"] >= (datetime.now() - timedelta(days=time_ranges[time_range]))]

st.subheader(f"📈 Pollutant Trends for {station} ({time_range})")
df_melt = df_trend.melt(
    id_vars=["date_aligned"],
    value_vars=selected_pollutants,
    var_name="Pollutant",
    value_name="Concentration"
)
if not df_melt.empty:
    fig_trend = px.line(df_melt, x="date_aligned", y="Concentration", color="Pollutant")
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.warning("⚠️ No pollutants selected to display.")

@st.cache_resource
def train_lstm_model(X_scaled, y_scaled, window_size, forecast_days):
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(window_size,1)),
        Dense(forecast_days)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y_scaled, epochs=8, batch_size=16, verbose=0)
    return model

def lstm_forecast(values, forecast_days):
    n_rows = len(values)
    if n_rows < 20:  # Not enough data
        return np.array([np.nan]*forecast_days), None

    window_size = min(30, n_rows - forecast_days)
    if window_size < 1:
        window_size = n_rows - 1

    X, y = [], []
    for i in range(n_rows - window_size - forecast_days + 1):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size:i+window_size+forecast_days])

    if not X:
        return np.array([np.nan]*forecast_days), None

    X, y = np.array(X), np.array(y)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1,1)).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1,1)).reshape(y.shape)

    model = train_lstm_model(X_scaled, y_scaled, window_size, forecast_days)
    last_seq = X_scaled[-1].reshape(1, window_size, 1)
    forecast_scaled = model.predict(last_seq, verbose=0).flatten()
    forecast_values = scaler.inverse_transform(forecast_scaled.reshape(-1,1)).flatten()
    return forecast_values, scaler

def retrain_and_forecast(df_station, station, forecast_days, forecast_label):
    values = df_station["AQI_Selected"].values.reshape(-1,1)
    forecast_values, _ = lstm_forecast(values, forecast_days)
    
    if forecast_values.size == 0:
        st.warning(f"Not enough data to forecast {forecast_label}.")
        return None
    
    today = datetime.now()
    if forecast_label == "24 Hours":
        dates = [today + timedelta(hours=i) for i in range(1, forecast_days+1)]
    else:
        dates = [today + timedelta(days=i) for i in range(1, forecast_days+1)]

    forecast_df = pd.DataFrame({"date": dates, "Forecast_AQI": forecast_values})
    
    # Save forecasts to viewable SQLite DB
    for d, aqi in zip(dates, forecast_values):
        c_viewable.execute(
            "INSERT INTO forecasts (station, forecast_label, forecast_date, forecast_aqi) VALUES (?,?,?,?)",
            (station, forecast_label, d.strftime("%Y-%m-%d %H:%M:%S"), float(aqi))
        )
    conn_viewable.commit()
    
    return forecast_df


# Automatic Forecast after Upload
forecast_df = retrain_and_forecast(df_station, station, forecast_days, forecast_label)
if forecast_df is not None:
    st.subheader(f"📈 {forecast_label} AQI Forecast (Auto-Retrained)")
    fig_forecast = px.line(forecast_df, x="date", y="Forecast_AQI",
                           labels={"date":"Date","Forecast_AQI":"AQI"},
                           title=f"AQI Forecast for {forecast_label}")
    fig_forecast.update_traces(mode='lines+markers', line=dict(color='firebrick'))
    st.plotly_chart(fig_forecast, use_container_width=True)


# Alerts
st.subheader("🚨 Alert Notifications")
def get_aqi_alerts(aqi):
    if aqi <= 50:
        return "Good ✅", ["Air quality is healthy. Enjoy outdoor activities!", "No precautions needed today."]
    elif aqi <= 100:
        return "Moderate 🙂", ["Air quality is satisfactory. Minor health concerns for sensitive groups.", "You can go outside, but monitor any irritation."]
    elif aqi <= 200:
        return "Unhealthy 😐", ["Air quality is moderate. People with respiratory issues should be careful.", "Consider limiting prolonged outdoor activity."]
    elif aqi <= 300:
        return "Very Unhealthy 😷", ["Air quality is poor. Sensitive groups should avoid outdoor activities.", "Use masks if you need to go outside."]
    elif aqi <= 400:
        return "Hazardous 🤢", ["Air quality is very poor. Everyone should reduce outdoor exposure.", "Close windows and use air purifiers if possible."]
    else:
        return "Severe ☠️", ["Air quality is severe. Avoid going outside completely.", "Health risks are very high — take all precautions."]

aqi_category, alerts = get_aqi_alerts(current_aqi)
for alert in alerts:
    if "Good" in aqi_category or "Moderate" in aqi_category:
        st.success(f"✅ {alert}")
    elif "Unhealthy" in aqi_category or "Very Unhealthy" in aqi_category:
        st.warning(f"⚠️ {alert}")
    else:
        st.error(f"❌ {alert}")

st.info(f"Current AQI: {current_aqi:.2f} → {aqi_category}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# ----------------------------
# Dashboard Title
# ----------------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("Air Quality Dashboard")

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("⚙️ Controls")

stations = ["Maharashtra", "Delhi", "Rajasthan", "Gujarat",
            "Tamil Nadu", "Karnataka", "West Bengal", "Uttar Pradesh"]

pollutants = ["PM2.5", "NO2", "O3", "CO", "PM10"]


station = st.sidebar.selectbox("Monitoring Station", stations)

pollutant = st.sidebar.selectbox("Pollutant", pollutants)


# ----------------------------
# Load Your Data
# ----------------------------
df = pd.read_csv(
    r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\city_hour.csv",
    parse_dates=["date"]
)


# ------------------------
# AQI Breakpoints
# ------------------------
breakpoints_dict = {
    "PM2.5": [
        {"low": 0, "high": 30, "index_low": 0, "index_high": 50},
        {"low": 31, "high": 60, "index_low": 51, "index_high": 100},
        {"low": 61, "high": 90, "index_low": 101, "index_high": 200},
        {"low": 91, "high": 120, "index_low": 201, "index_high": 300},
        {"low": 121, "high": 250, "index_low": 301, "index_high": 400},
        {"low": 251, "high": 500, "index_low": 401, "index_high": 500}
    ],
    "PM10": [
        {"low": 0, "high": 50, "index_low": 0, "index_high": 50},
        {"low": 51, "high": 100, "index_low": 51, "index_high": 100},
        {"low": 101, "high": 250, "index_low": 101, "index_high": 200},
        {"low": 251, "high": 350, "index_low": 201, "index_high": 300},
        {"low": 351, "high": 430, "index_low": 301, "index_high": 400},
        {"low": 431, "high": 1000, "index_low": 401, "index_high": 500}
    ],
    "NO2": [
        {"low": 0, "high": 40, "index_low": 0, "index_high": 50},
        {"low": 41, "high": 80, "index_low": 51, "index_high": 100},
        {"low": 81, "high": 180, "index_low": 101, "index_high": 200},
        {"low": 181, "high": 280, "index_low": 201, "index_high": 300},
        {"low": 281, "high": 400, "index_low": 301, "index_high": 400},
        {"low": 401, "high": 1000, "index_low": 401, "index_high": 500}
    ],
    "CO": [
        {"low": 0, "high": 1, "index_low": 0, "index_high": 50},
        {"low": 1.1, "high": 2, "index_low": 51, "index_high": 100},
        {"low": 2.1, "high": 10, "index_low": 101, "index_high": 200},
        {"low": 10.1, "high": 17, "index_low": 201, "index_high": 300},
        {"low": 17.1, "high": 34, "index_low": 301, "index_high": 400},
        {"low": 34.1, "high": 50, "index_low": 401, "index_high": 500}
    ],
    "O3": [
        {"low": 0, "high": 50, "index_low": 0, "index_high": 50},
        {"low": 51, "high": 100, "index_low": 51, "index_high": 100},
        {"low": 101, "high": 168, "index_low": 101, "index_high": 200},
        {"low": 169, "high": 208, "index_low": 201, "index_high": 300},
        {"low": 209, "high": 748, "index_low": 301, "index_high": 400},
        {"low": 749, "high": 1000, "index_low": 401, "index_high": 500}
    ]
}

def calculate_aqi_subindex(concentration, breakpoints):
    for bp in breakpoints:
        if bp["low"] <= concentration <= bp["high"]:
            Clow, Chigh = bp["low"], bp["high"]
            Ilow, Ihigh = bp["index_low"], bp["index_high"]
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (concentration - Clow) + Ilow
    return None

# ============================
# SECTION 1: Gauge
# ============================
st.subheader(f"Current {pollutant} AQI ({station})")

df_range = df[df["State"] == station].copy()


def safe_aqi(row, pollutant):
    val = row[pollutant]
    if pd.isna(val):
        return 0
    return calculate_aqi_subindex(val, breakpoints_dict[pollutant])

df_range["AQI_Selected"] = df_range.apply(lambda row: safe_aqi(row, pollutant), axis=1)
current_aqi = df_range["AQI_Selected"].iloc[-1] if not df_range.empty else 0

if current_aqi <= 50:
    aqi_category, color = "Good", "green"
elif current_aqi <= 100:
    aqi_category, color = "Moderate", "yellow"
elif current_aqi <= 200:
    aqi_category, color = "Unhealthy", "orange"
elif current_aqi <= 300:
    aqi_category, color = "Very Unhealthy", "red"
else:
    aqi_category, color = "Hazardous", "maroon"

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

# ============================
# SECTION 2: Pollutant Trend
# ============================
st.subheader(f"{station} - {pollutant} Trend")

df_state = df[df['State'] == station].sort_values("date")
if not df_state.empty:
    fig_trend = px.line(df_state, x="date", y=pollutant,
                        labels={"value": "Concentration", "date": "Date"})
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.warning(f"No data available for {station}.")

# ============================
# SECTION 3: Forecast
# ============================
st.subheader("7-Day AQI Forecast")

forecast_days = 7
values = df_state["AQI"].dropna().values.reshape(-1, 1)
n_rows = len(values)

def get_aqi_color(aqi):
    if aqi <= 50:
        return '#d5f6e3'
    elif aqi <= 100:
        return '#f9f7d8'
    elif aqi <= 150:
        return '#fff3cd'
    else:
        return '#fbe7c6'

if n_rows >= 2:
    window_size = min(30, n_rows - 1)
    X, y = [], []
    for i in range(n_rows - window_size):
        X.append(values[i:i + window_size])
        y.append(values[i + window_size])

    if len(X) > 0:
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

        forecast_values = scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)).flatten()
        forecast_dates = [(datetime.today() + timedelta(days=i)).strftime('%a')
                          for i in range(1, forecast_days + 1)]

        cols_forecast = st.columns(forecast_days)
        for i, col in enumerate(cols_forecast):
            color = get_aqi_color(forecast_values[i])
            col.markdown(
                f"<div style='background:{color}; padding:12px; border-radius:12px; text-align:center;'>"
                f"<strong>{forecast_dates[i]}</strong><br>AQI {forecast_values[i]:.2f}"
                f"</div>", unsafe_allow_html=True
            )
    else:
        st.warning("Not enough sequences for LSTM training.")
else:
    st.warning("Not enough data to forecast AQI.")

# ============================
# SECTION 4: Alerts
# ============================
st.subheader("🚨 Alert Notifications")

def get_aqi_alerts(aqi):
    if aqi <= 50:
        return "Good ✅", [
            "Air quality is healthy. Enjoy outdoor activities!",
            "No precautions needed today."
        ]
    elif aqi <= 100:
        return "Moderate 🙂", [
            "Air quality is satisfactory. Minor health concerns for sensitive groups.",
            "You can go outside, but monitor any irritation."
        ]
    elif aqi <= 200:
        return "Unhealthy 😐", [
            "Air quality is moderate. People with respiratory issues should be careful.",
            "Consider limiting prolonged outdoor activity."
        ]
    elif aqi <= 300:
        return "Very Unhealthy 😷", [
            "Air quality is poor. Sensitive groups should avoid outdoor activities.",
            "Use masks if you need to go outside."
        ]
    elif aqi <= 400:
        return "Hazardous 🤢", [
            "Air quality is very poor. Everyone should reduce outdoor exposure.",
            "Close windows and use air purifiers if possible."
        ]
    else:
        return "Severe ☠️", [
            "Air quality is severe. Avoid going outside completely.",
            "Health risks are very high — take all precautions."
        ]

aqi_category, alerts = get_aqi_alerts(current_aqi)
for alert in alerts:
    if "Good" in aqi_category or "Moderate" in aqi_category:
        st.success(f"✅ {alert}")
    elif "Unhealthy" in aqi_category or "Very Unhealthy" in aqi_category:
        st.warning(f"⚠️ {alert}")
    else:
        st.error(f"❌ {alert}")

st.info(f"Current AQI: {current_aqi:.2f} → {aqi_category}")

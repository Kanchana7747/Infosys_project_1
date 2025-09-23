import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
import matplotlib.pyplot as plt

# ------------------------
# 1. Load Data
# ------------------------


df = pd.read_csv(r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\air_quality_milestone1.csv", parse_dates=["date"])
df.columns = df.columns.str.strip()  # remove any leading/trailing spaces
print(df.columns)
city = st.selectbox("Select City", df["station_id"].unique())
pollutant = st.selectbox("Select Pollutant", ["PM2.5", "PM10", "NO2", "CO", "O3"])

# Filter data for selected city
city_df = df[df["station_id"] == city].copy()
# ------------------------
# 3. AQI Breakpoints for pollutants
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
        {"low": 431, "high": 600, "index_low": 401, "index_high": 500}
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

# ------------------------
# 4. AQI Sub-index calculation
# ------------------------
def calculate_aqi_subindex(concentration, breakpoints):
    for bp in breakpoints:
        if bp["low"] <= concentration <= bp["high"]:
            Clow, Chigh = bp["low"], bp["high"]
            Ilow, Ihigh = bp["index_low"], bp["index_high"]
            return ((Ihigh - Ilow)/(Chigh - Clow))*(concentration - Clow) + Ilow
    return None

# ------------------------
# 5. Load Prophet model
# ------------------------
model = joblib.load("best_model.pkl")  # make sure it's trained for the pollutant

# Prepare data for Prophet
df_prophet = city_df[["date", pollutant]].rename(columns={"date": "ds", pollutant: "y"})

# Forecast next 7 days
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)
future_values = forecast["yhat"].iloc[-7:].values

# Convert to AQI
future_aqi = [np.round(calculate_aqi_subindex(val, breakpoints_dict[pollutant]), 2) for val in future_values]

# ------------------------
# 6. Risk level function
# ------------------------
def risk_level(aqi):
    if aqi <= 50: return "Good ✅"
    elif aqi <= 100: return "Satisfactory 🙂"
    elif aqi <= 200: return "Moderate 😐"
    elif aqi <= 300: return "Poor 😷"
    elif aqi <= 400: return "Very Poor 🤢"
    else: return "Severe ☠️"

# ------------------------
# 7. Display results in Streamlit
# ------------------------
st.write(f"### AQI Forecast for {city} ({pollutant}) for Next 7 Days")
for i, val in enumerate(future_aqi, 1):
    st.write(f"Day {i}: AQI {val} → {risk_level(val)}")

# Optional: Circular visualization for last 3 days
fig, ax = plt.subplots(figsize=(4,4))
colors = ["green","yellow","orange","red","purple","black"]
labels = [risk_level(val) for val in future_aqi[-3:]]
sizes = [1,1,1]  # equal slices
ax.pie(sizes, labels=labels, colors=colors[:3], startangle=90)
ax.set_title("Next 3 Days Risk Levels")
st.pyplot(fig)

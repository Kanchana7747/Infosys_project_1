import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from prophet import Prophet

# ------------------------
# 1. AQI Sub-index Calculation
# ------------------------
def calculate_aqi_subindex(concentration, breakpoints):
    for bp in breakpoints:
        if bp["low"] <= concentration <= bp["high"]:
            Clow, Chigh = bp["low"], bp["high"]
            Ilow, Ihigh = bp["index_low"], bp["index_high"]
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (concentration - Clow) + Ilow
    return None

# Example breakpoints dictionary for all pollutants
breakpoints_dict = {
    "PM2.5": [
        {"low":0,"high":30,"index_low":0,"index_high":50},
        {"low":31,"high":60,"index_low":51,"index_high":100},
        {"low":61,"high":90,"index_low":101,"index_high":200},
        {"low":91,"high":120,"index_low":201,"index_high":300},
        {"low":121,"high":250,"index_low":301,"index_high":400},
        {"low":251,"high":500,"index_low":401,"index_high":500}
    ],
    "PM10": [
        {"low":0,"high":50,"index_low":0,"index_high":50},
        {"low":51,"high":100,"index_low":51,"index_high":100},
        {"low":101,"high":250,"index_low":101,"index_high":200},
        {"low":251,"high":350,"index_low":201,"index_high":300},
        {"low":351,"high":430,"index_low":301,"index_high":400},
        {"low":431,"high":1000,"index_low":401,"index_high":500}
    ],
    "NO2": [
        {"low":0,"high":40,"index_low":0,"index_high":50},
        {"low":41,"high":80,"index_low":51,"index_high":100},
        {"low":81,"high":180,"index_low":101,"index_high":200},
        {"low":181,"high":280,"index_low":201,"index_high":300},
        {"low":281,"high":400,"index_low":301,"index_high":400},
        {"low":401,"high":1000,"index_low":401,"index_high":500}
    ],
    "CO": [
        {"low":0,"high":1,"index_low":0,"index_high":50},
        {"low":1.1,"high":2,"index_low":51,"index_high":100},
        {"low":2.1,"high":10,"index_low":101,"index_high":200},
        {"low":10.1,"high":17,"index_low":201,"index_high":300},
        {"low":17.1,"high":34,"index_low":301,"index_high":400},
        {"low":34.1,"high":50,"index_low":401,"index_high":500}
    ],
    "O3": [
        {"low":0,"high":50,"index_low":0,"index_high":50},
        {"low":51,"high":100,"index_low":51,"index_high":100},
        {"low":101,"high":168,"index_low":101,"index_high":200},
        {"low":169,"high":208,"index_low":201,"index_high":300},
        {"low":209,"high":748,"index_low":301,"index_high":400},
        {"low":749,"high":1000,"index_low":401,"index_high":500}
    ]
}

# ------------------------
# 2. Compute AQI for each row
# ------------------------
def compute_overall_aqi(row):
    sub_indices = []
    for pollutant, breakpoints in breakpoints_dict.items():
        sub_indices.append(calculate_aqi_subindex(row[pollutant], breakpoints))
    return max(sub_indices)

# ------------------------
# 3. Load Data
# ------------------------
df = pd.read_csv(r"C:\Users\slaxm\OneDrive\Documents\air_quality_milestone1.csv", parse_dates=["date"])
df["AQI"] = df.apply(compute_overall_aqi, axis=1)

# ------------------------
# 4. Forecast next 7 days for each pollutant using Prophet
# ------------------------
future_aqi_dict = {}

for pollutant in ["PM2.5","PM10","NO2","CO","O3"]:
    prophet_model = Prophet()
    df_prophet = df[["date", pollutant]].rename(columns={"date":"ds", pollutant:"y"})
    prophet_model.fit(df_prophet)
    
    future = prophet_model.make_future_dataframe(periods=7)
    forecast = prophet_model.predict(future)
    
    future_vals = forecast["yhat"].iloc[-7:].values
    future_aqi_dict[pollutant] = [np.round(calculate_aqi_subindex(val, breakpoints_dict[pollutant]),2) 
                                   for val in future_vals]

# ------------------------
# 5. Risk Level Function
# ------------------------
def risk_level(aqi):
    if aqi <= 50: return "Good ✅"
    elif aqi <= 100: return "Satisfactory 🙂"
    elif aqi <= 200: return "Moderate 😐"
    elif aqi <= 300: return "Poor 😷"
    elif aqi <= 400: return "Very Poor 🤢"
    else: return "Severe ☠️"

# ------------------------
# 6. Print Forecast AQI for all pollutants
# ------------------------
for pollutant, values in future_aqi_dict.items():
    print(f"\n📅 Forecast for {pollutant}:")
    for i, val in enumerate(values,1):
        print(f"Day {i}: AQI {val} → {risk_level(val)}")

# ------------------------
# 7. Plot Last 30 Days AQI + Forecast for selected pollutant
# ------------------------
selected_pollutant = "PM2.5"  # Can make this dynamic in Streamlit

plt.figure(figsize=(10,5))
plt.plot(df["date"].iloc[-30:], df[selected_pollutant].iloc[-30:], label=f"Last 30 Days {selected_pollutant}")
plt.plot(pd.date_range(df["date"].iloc[-1], periods=8, freq="D")[1:], future_aqi_dict[selected_pollutant],
         label=f"Forecast {selected_pollutant} AQI (Next 7 Days)", linestyle="--", marker="o")

plt.axhline(50, color="green", linestyle="--", alpha=0.5)
plt.axhline(100, color="yellow", linestyle="--", alpha=0.5)
plt.axhline(200, color="orange", linestyle="--", alpha=0.5)
plt.axhline(300, color="red", linestyle="--", alpha=0.5)

plt.title(f"{selected_pollutant} Forecast AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.show()

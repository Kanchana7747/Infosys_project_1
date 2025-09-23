import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\city_hour.csv", parse_dates=["date"])

# Clean column names and state names
df.columns = df.columns.str.strip()
df['State'] = df['State'].str.strip()

# -----------------------------
# Streamlit Layout
# -----------------------------
st.set_page_config(page_title="Air Quality Line Plot", layout="wide")
st.title("🌫️ Air Quality Line Plot")

# -----------------------------
# User selects state and pollutants
# -----------------------------
states = df['State'].unique()
pollutants = ["AQI", "NO2", "O3", "CO", "PM2.5", "PM10"]

selected_state = st.selectbox("Select State:", states)
selected_pollutants = st.multiselect("Select Pollutants:", pollutants, default=["AQI", "PM2.5"])

# Filter data
df_state = df[df['State'] == selected_state].sort_values("date")

# -----------------------------
# Plot line chart
# -----------------------------
if df_state.empty:
    st.warning(f"No data available for {selected_state}.")
else:
    fig = px.line(df_state, x="date", y=selected_pollutants,
                  title=f"{selected_state} Pollutant Trends",
                  labels={"value":"Concentration", "date":"Date"})
    st.plotly_chart(fig, use_container_width=True)

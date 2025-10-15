import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Import your other pages if needed
# import forecast_visualization
# import alerts_and_warnings
# import trend_analysis
# import admin_panel

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'> Air Quality Dashboard</h1>", unsafe_allow_html=True)

# ----------------------------
# SQLite setup for forecasts (viewable in VS Code)
# ----------------------------
DB_PATH = r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\aqi_forecasts_viewable.db"
conn_viewable = sqlite3.connect(DB_PATH, check_same_thread=False)
c_viewable = conn_viewable.cursor()

# AQI Forecast table
c_viewable.execute("""
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station TEXT,
    forecast_label TEXT,
    forecast_date TEXT,
    forecast_aqi REAL
)
""")

# Pollutant Forecast table
c_viewable.execute("""
CREATE TABLE IF NOT EXISTS pollutant_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station TEXT,
    pollutant TEXT,
    forecast_label TEXT,
    forecast_date TEXT,
    forecast_value REAL
)
""")
conn_viewable.commit()

# ----------------------------
# Load Default Dataset
# ----------------------------
def load_default_data():
    return pd.read_csv(r"city_day.csv")
df=load_default_data()
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

# ----------------------------
# Sidebar Navigation
# ----------------------------
with st.sidebar:
    app = option_menu(
        menu_title="Navigation",
        options=["Home", "Forecast Visualization", "Trend Analysis", "Alerts and Warnings", "Admin Panel"],
        icons=["house", "bar-chart-line", "graph-up-arrow", "bell", "person-badge"],
        menu_icon="cast",
        default_index=0,
    )

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
location_col = next((c for c in ["State", "station_id", "City", "StationId"] if c in df.columns), None)
if not location_col:
    st.error("Dataset must have a location column: 'State', 'station_id', or 'City'.")
    st.stop()
st.session_state["location_col"] = location_col

stations = df[location_col].dropna().unique()
time_ranges = {"Last 7 Days": 7, "Last 30 Days": 30}
pollutants = ["PM2.5", "NO2", "O3", "CO", "PM10", "SO2"]
forecast_options = {"24 Hours": "hourly", "7 Days": "daily", "15 Days": "daily"}

station = st.sidebar.selectbox(f"Select {location_col}", stations)
time_range = st.sidebar.selectbox("Time Range", list(time_ranges.keys()))
selected_pollutants = st.sidebar.multiselect("Select pollutants to display", pollutants)
forecast_label = st.sidebar.selectbox("Forecast Horizon", list(forecast_options.keys()))
forecast_days = {"24 Hours": 24, "7 Days": 7, "15 Days": 15}[forecast_label]

# Filter Data
end_date = df["date"].max()
start_date = end_date - pd.Timedelta(days=time_ranges[time_range])
df_station = df[df[location_col] == station].sort_values("date")

# ----------------------------
# Data Cleaning and AQI Calculation
# ----------------------------
for col in pollutants + ["AQI"]:
    if col in df_station.columns:
        df_station[col].fillna(df_station[col].median(), inplace=True)
        Q1 = df_station[col].quantile(0.25)
        Q3 = df_station[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_station[col] = df_station[col].clip(lower=lower_bound, upper=upper_bound)

common_breakpoints = [
    {"low": 0, "high": 50, "index_low": 0, "index_high": 50},
    {"low": 51, "high": 100, "index_low": 51, "index_high": 100},
    {"low": 101, "high": 200, "index_low": 101, "index_high": 200},
    {"low": 201, "high": 300, "index_low": 201, "index_high": 300},
    {"low": 301, "high": 400, "index_low": 301, "index_high": 400},
    {"low": 401, "high": 500, "index_low": 401, "index_high": 500}
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
# Home Page
# ----------------------------
if app == "Home":
    st.title("🌍 Air Aware - Air Quality Prediction Dashboard")
    st.write("Welcome to Air Aware! Use the filters in the sidebar to explore AQI trends, forecasts, and warnings.")
    st.write(df.head())

    # AQI Gauge
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

# ----------------------------
# Forecast Visualization
# ----------------------------
elif app == "Forecast Visualization":
    st.title("📊 Forecast Visualization")
    df_trend = df[df[st.session_state["location_col"]] == station].copy()
    df_trend["date"] = pd.to_datetime(df_trend["date"])
    latest_data_date = df_trend["date"].max()
    shift_days = (datetime.now().date() - latest_data_date.date()).days
    df_trend["date_aligned"] = df_trend["date"] + pd.to_timedelta(shift_days, unit="D")
    df_trend = df_trend[df_trend["date_aligned"] >= (datetime.now() - timedelta(days=time_ranges[time_range]))]

    if not df_trend.empty:
        inferred_freq = pd.infer_freq(df_trend["date_aligned"].sort_values())
        freq = "H" if inferred_freq and "H" in inferred_freq else "D"
        date_range = pd.date_range(df_trend["date_aligned"].min(), df_trend["date_aligned"].max(), freq=freq)
        df_trend = df_trend.set_index("date_aligned").reindex(date_range).rename_axis("date_aligned").reset_index()
        for pol in selected_pollutants:
            if pol in df_trend.columns:
                median_val = df_trend[pol].median()
                df_trend[pol].fillna(median_val, inplace=True)

    st.subheader(f"📈 Pollutant Trends for {station} ({time_range})")
    df_melt = df_trend.melt(
        id_vars=["date_aligned"],
        value_vars=[pol for pol in selected_pollutants if pol in df_trend.columns],
        var_name="Pollutant",
        value_name="Concentration"
    )
    if not df_melt.empty:
        fig_trend = px.line(df_melt, x="date_aligned", y="Concentration", color="Pollutant")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("⚠️ No pollutants selected to display.")

    # ----------------------------
    # LSTM Forecast Functions
    # ----------------------------
    @st.cache_resource
    def train_lstm_model(X_scaled, y_scaled, window_size, forecast_days):
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(window_size, 1)),
            Dense(forecast_days)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_scaled, y_scaled, epochs=8, batch_size=16, verbose=0)
        return model

    def lstm_forecast(values, forecast_days):
        n_rows = len(values)
        if n_rows < 20:
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

    # ----------------------------
    # Forecast & Pollutant Forecast Functions
    # ----------------------------
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
        if c_viewable:
            for d, aqi in zip(dates, forecast_values):
                c_viewable.execute(
                    "INSERT INTO forecasts (station, forecast_label, forecast_date, forecast_aqi) VALUES (?,?,?,?)",
                    (station, forecast_label, d.strftime("%Y-%m-%d %H:%M:%S"), float(aqi))
                )
            conn_viewable.commit()
        return forecast_df

    forecast_df = retrain_and_forecast(df_station, station, forecast_days, forecast_label)
    if forecast_df is not None:
        st.subheader(f"📈 {forecast_label} AQI Forecast (Auto-Retrained)")
        fig_forecast = px.line(forecast_df, x="date", y="Forecast_AQI",
                            labels={"date":"Date","Forecast_AQI":"AQI"},
                            title=f"AQI Forecast for {forecast_label}")
        fig_forecast.update_traces(mode='lines+markers', line=dict(color='firebrick'))
        st.plotly_chart(fig_forecast, use_container_width=True)
    st.session_state["forecast_df"] = forecast_df

    def forecast_pollutants(df_station, station, forecast_days, forecast_label, pollutants):
        forecast_results = {}
        today = datetime.now()
        for pol in pollutants:
            if pol not in df_station.columns:
                continue
            values = df_station[pol].values.reshape(-1,1)
            forecast_values, _ = lstm_forecast(values, forecast_days)
            if forecast_values.size == 0:
                continue
            if forecast_label == "24 Hours":
                dates = [today + timedelta(hours=i) for i in range(1, forecast_days+1)]
            else:
                dates = [today + timedelta(days=i) for i in range(1, forecast_days+1)]
            if c_viewable:
                for d, val in zip(dates, forecast_values):
                    c_viewable.execute(
                        "INSERT INTO pollutant_forecasts (station, pollutant, forecast_label, forecast_date, forecast_value) VALUES (?,?,?,?,?)",
                        (station, pol, forecast_label, d.strftime("%Y-%m-%d %H:%M:%S"), float(val))
                    )
                conn_viewable.commit()
            forecast_results[pol] = pd.DataFrame({"date": dates, "Forecast": forecast_values})
        return forecast_results

    pollutant_forecast_results = forecast_pollutants(df_station, station, forecast_days, forecast_label, selected_pollutants)
    for pol, df_forecast in pollutant_forecast_results.items():
        st.subheader(f"📈 {pol} Forecast for {forecast_label}")
        fig_pol = px.line(df_forecast, x="date", y="Forecast",
                        labels={"date":"Date", "Forecast":pol},
                        title=f"{pol} Forecast")
        fig_pol.update_traces(mode='lines+markers')
        st.plotly_chart(fig_pol, use_container_width=True)

# ----------------------------
# Trend Analysis Page
# ----------------------------
elif app == "Trend Analysis":
    st.title(f"📊 Trend Analysis for {station}")

    # Filter data for the selected station
    df_station = df[df[location_col] == station].copy()
    df_station["date"] = pd.to_datetime(df_station["date"])
    df_station.sort_values("date", inplace=True)

    # Ensure AQI_Selected exists
    if "AQI_Selected" not in df_station.columns:
        df_station["AQI_Selected"] = df_station["AQI"] if "AQI" in df_station.columns else 0

    # -----------------------------
    # 1️⃣ Yearly Heatmap of Pollutants
    # -----------------------------
    st.subheader("🌡️ Yearly Heatmap of Pollutants")
    df_station["year"] = df_station["date"].dt.year

    if selected_pollutants:
        for pol in selected_pollutants:
            if pol in df_station.columns:
                heat_data = df_station.groupby("year")[pol].mean().to_frame()
                fig_heat = px.imshow(
                    heat_data.T,
                    labels=dict(x="Year", y="Pollutant", color=f"{pol} Avg Concentration"),
                    title=f"{pol} Yearly Average Concentration"
                )
                st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("No pollutants selected to display heatmap.")

    # -----------------------------
    # 2️⃣ Pollutant Contribution Analysis
    # -----------------------------
    st.subheader("📊 Pollutants Contribution Analysis")
    if "AQI_Selected" not in df_station.columns:
        st.warning("AQI_Selected column not found. Cannot compute contribution.")
    else:
        df_contrib = df_station[selected_pollutants].copy()
        df_contrib["AQI_Selected"] = df_station["AQI_Selected"].replace(0, np.nan)

        # Normalize pollutant contribution to AQI
        for pol in selected_pollutants:
            df_contrib[pol] = df_contrib[pol] / df_contrib["AQI_Selected"]

        avg_contrib = df_contrib[selected_pollutants].mean().fillna(0)

        # Bar chart
        fig_bar = px.bar(
            x=avg_contrib.index,
            y=avg_contrib.values,
            labels={"x": "Pollutant", "y": "Average Contribution"},
            title=f"Average Contribution of Pollutants to AQI at {station}"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie chart
        fig_pie = px.pie(
            values=avg_contrib.values,
            names=avg_contrib.index,
            title=f"Pollutants Contribution to AQI at {station}"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------
    # 3️⃣ Seasonal Decomposition of AQI
    # -----------------------------
    st.subheader("📈 Seasonal Decomposition of AQI")
    # Prepare time series
    ts = df_station.set_index("date")["AQI_Selected"]

    # Resample daily and fill missing values
    ts = ts.asfreq("D")
    ts = ts.fillna(method="ffill").fillna(method="bfill")

    # Limit to last 6 months
    six_months_ago = ts.index.max() - pd.DateOffset(months=6)
    ts_recent = ts[ts.index >= six_months_ago]

    # Check if enough data
    if len(ts_recent) >= 30:  # minimum 1 month for decomposition
        decomposition = seasonal_decompose(ts_recent, model="additive", period=30)  # monthly seasonality

        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Scatter(x=ts_recent.index, y=decomposition.trend, mode="lines", name="Trend"))
        fig_decomp.add_trace(go.Scatter(x=ts_recent.index, y=decomposition.seasonal, mode="lines", name="Seasonal"))
        fig_decomp.add_trace(go.Scatter(x=ts_recent.index, y=decomposition.resid, mode="lines", name="Residual"))

        fig_decomp.update_layout(
            title=f"Seasonal Decomposition of AQI (Last 6 Months) for {station}",
            xaxis_title="Date",
            yaxis_title="AQI"
        )
        st.plotly_chart(fig_decomp, use_container_width=True)
    else:
        st.warning("Not enough data for seasonal decomposition. At least 1 month of daily data is required.")



# ----------------------------
# Alerts and Warnings Page
# ----------------------------

elif app == "Alerts and Warnings":
    st.subheader("🚨 Alert Notifications")

    # 1️⃣ Ensure forecast_df is available
    forecast_df = st.session_state.get("forecast_df", None)

    # 2️⃣ Define AQI alert messages
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

    # 3️⃣ Create SQLite table for alerts (if not exists)
    c_viewable.execute("""
        CREATE TABLE IF NOT EXISTS aqi_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT,
            alert_date TEXT,
            aqi_value REAL,
            category TEXT,
            message TEXT
        )
    """)
    conn_viewable.commit()

    # 4️⃣ Display and store current AQI alerts
    aqi_category, alerts = get_aqi_alerts(current_aqi)
    st.info(f"Current AQI: {current_aqi:.2f} → {aqi_category}")

    for alert in alerts:
        if "Good" in aqi_category or "Moderate" in aqi_category:
            st.success(f"✅ {alert}")
        elif "Unhealthy" in aqi_category or "Very Unhealthy" in aqi_category:
            st.warning(f"⚠️ {alert}")
        else:
            st.error(f"❌ {alert}")

        # Insert into DB only if it doesn't already exist
        c_viewable.execute("""
            SELECT COUNT(*) FROM aqi_alerts
            WHERE station=? AND alert_date=? AND message=?
        """, (station, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alert))
        if c_viewable.fetchone()[0] == 0:
            c_viewable.execute("""
                INSERT INTO aqi_alerts (station, alert_date, aqi_value, category, message)
                VALUES (?, ?, ?, ?, ?)
            """, (station, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_aqi, aqi_category, alert))
    conn_viewable.commit()

    # 5️⃣ Display and store forecast-based alerts
    if forecast_df is not None and not forecast_df.empty:
        st.markdown("### Forecasted AQI Alerts")
        for d, aqi in zip(forecast_df["date"].values, forecast_df["Forecast_AQI"].values):
            category, messages = get_aqi_alerts(aqi)
            st.markdown(f"**{pd.to_datetime(d).strftime('%Y-%m-%d %H:%M')} → AQI {aqi:.1f} ({category})**")
            for msg in messages:
                if "Good" in category or "Moderate" in category:
                    st.success(f"✅ {msg}")
                elif "Unhealthy" in category or "Very Unhealthy" in category:
                    st.warning(f"⚠️ {msg}")
                else:
                    st.error(f"❌ {msg}")

                # Insert forecast alert into DB
                c_viewable.execute("""
                    SELECT COUNT(*) FROM aqi_alerts
                    WHERE station=? AND alert_date=? AND message=?
                """, (station, pd.to_datetime(d).strftime("%Y-%m-%d %H:%M:%S"), msg))
                if c_viewable.fetchone()[0] == 0:
                    c_viewable.execute("""
                        INSERT INTO aqi_alerts (station, alert_date, aqi_value, category, message)
                        VALUES (?, ?, ?, ?, ?)
                    """, (station, pd.to_datetime(d).strftime("%Y-%m-%d %H:%M:%S"), float(aqi), category, msg))
        conn_viewable.commit()
    else:
        st.warning("⚠️ No forecast data available. Generate AQI forecast first.")


# ----------------------------
# Admin Panel Page
# ----------------------------
elif app == "Admin Panel":
    st.title("🛠️ Admin Panel")
    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False

    # --- Admin Login ---
    if not st.session_state["admin_logged_in"]:
        st.subheader("Admin Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "admin@123":
                st.session_state["admin_logged_in"] = True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials")
        st.stop()

    # --- Upload Dataset ---
    st.subheader("Upload Dataset (CSV or Excel)")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df_new = pd.read_csv(uploaded_file)
        else:
            df_new = pd.read_excel(uploaded_file)

        # Standardize column names
        df_new.columns = df_new.columns.str.strip()
        df_new.rename(columns={"Datetime":"date","Date":"date","StationId":"station_id","City":"station_id"}, inplace=True)

        # Check required columns
        required_cols = ["date", "station_id"]
        missing_cols = [col for col in required_cols if col not in df_new.columns]
        if missing_cols:
            st.error(f"Dataset must contain: {', '.join(missing_cols)}")
            st.stop()

        # Preview dataset
        st.write("Dataset preview (first 10 rows):")
        st.dataframe(df_new.head(10))

        # --- Append to Database ---
    st.subheader("⚡ Append to Database")
    if st.button("Append to Database"):
        try:
            # Connect to DB and create cursor
            conn = sqlite3.connect(r"C:\Users\slaxm\OneDrive\Documents\Air_Aware\city_day.db")
            c = conn.cursor()

            # Rename column to avoid dot
            if "PM2.5" in df_new.columns:
                df_new.rename(columns={"PM2.5": "PM25"}, inplace=True)

            # Keep only allowed columns
            allowed_cols = ["date","station_id","AQI","PM25","PM10","NO2","SO2","O3","CO"]
            df_new = df_new[[col for col in df_new.columns if col in allowed_cols]]
            df_new = df_new.reindex(columns=allowed_cols, fill_value=None)

            # Create table if not exists
            c.execute("""
                CREATE TABLE IF NOT EXISTS city_data (
                    date TEXT,
                    station_id TEXT,
                    AQI REAL,
                    PM25 REAL,
                    PM10 REAL,
                    NO2 REAL,
                    SO2 REAL,
                    O3 REAL,
                    CO REAL
                )
            """)

            # Prepare data for insertion
            data_tuples = [tuple(x) for x in df_new.to_numpy()]
            columns = ','.join(df_new.columns)
            placeholders = ','.join('?' for _ in df_new.columns)

            # Insert data
            c.executemany(f"INSERT OR REPLACE INTO city_data ({columns}) VALUES ({placeholders})", data_tuples)

            # Commit and close connection
            conn.commit()
            conn.close()

            st.success(f"✅ Dataset appended successfully! Total rows: {len(df_new)}")
            st.info("You can now go to the Forecast tab to retrain the model with the updated dataset.")
            
        except Exception as e:
            st.error(f"Error while inserting data: {e}")











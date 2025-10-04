**Air Aware – Air Quality Monitoring & Forecasting**
**Project Overview**
A Python Streamlit dashboard for monitoring air quality and forecasting pollutants.
Allows data upload, trend analysis, and forecasting.
Uses SQLite for storing existing and forecasted data.
Forecasting is done using LSTM models for AQI and pollutants.

🛠 Features
**1. User Dashboard**
View trends of pollutants (PM2.5, PM10, NO2, SO2, O3, CO).
Check AQI trends over time.
Generate seasonal decomposition plots of AQI.
Create heatmaps and pollutant contribution analysis.
Forecast next days’ air quality.

**2. Admin Panel**
Secure login (username: admin, password: admin@123).
Upload new CSV/Excel datasets.
Append data to existing SQLite database.
Retrain forecast models by clicking on the Forecast tab.
Preview data and basic visualizations before appending.

**🗄 Database****
SQLite used to store pollutant measurements and forecast results.
Table city_data columns:
date, station_id, AQI, PM25, PM10, NO2, SO2, O3, CO.


**🔄 Updating Data & Retraining**
Upload datasets through Admin Panel.
Click Forecast tab to retrain models and update predictions.
Forecast results are saved in SQLite.

**📊 Visualizations**
Yearly heatmaps for selected pollutants.
Seasonal decomposition of AQI.
Average pollutant contribution to AQI.
Forecast plots for selected locations.





Do you want me to do that?

ChatGPT can make mi

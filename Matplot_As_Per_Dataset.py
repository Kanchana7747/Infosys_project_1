import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\slaxm\OneDrive\Documents\air_quality_milestone1.csv")  # replace with your dataset

# Ensure Date column is datetime
df['date'] = pd.to_datetime(df['date'])

# 1. LINE PLOT - Trend of PM2.5 over time
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['PM2.5'], color='blue', label="PM2.5")
plt.xlabel("Date")
plt.ylabel("PM2.5 concentration (µg/m³)")
plt.title("Trend of PM2.5 Over Time")
plt.legend()
plt.grid(True)
plt.show()

# 2. SCATTER PLOT - PM2.5 vs AQI
plt.figure(figsize=(7, 5))
plt.scatter(df['PM2.5'], df['AQI'], alpha=0.6, color="red")
plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("AQI")
plt.title("Scatter Plot - PM2.5 vs AQI")
plt.show()

# 3. BAR CHART - Average pollutant levels
avg_pollutants = df[['PM2.5', 'PM10', 'NO2', 'CO', 'O3']].mean()
plt.figure(figsize=(8, 5))
avg_pollutants.plot(kind='bar', color=['blue', 'orange', 'green', 'red', 'purple'])
plt.ylabel("Average Concentration (µg/m³)")
plt.title("Average Pollutant Levels")
plt.show()

# 4. HISTOGRAM - AQI distribution
plt.figure(figsize=(8, 5))
plt.hist(df['AQI'], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.title("Histogram of AQI Distribution")
plt.show()

# 5. BOXPLOT - Detect outliers in pollutants
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['PM2.5', 'PM10', 'NO2', 'CO', 'O3']])
plt.title("Boxplot of Pollutants (Outlier Detection)")
plt.show()

# 6. SUBPLOTS - PM2.5 and AQI trends side by side
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(df['date'], df['PM2.5'], color='blue')
axs[0].set_title("PM2.5 Over Time")
axs[0].set_ylabel("PM2.5")

axs[1].plot(df['date'], df['AQI'], color='red')
axs[1].set_title("AQI Over Time")
axs[1].set_ylabel("AQI")

plt.tight_layout()
plt.show()

# 7. HEATMAP - Correlation between pollutants & AQI
plt.figure(figsize=(8, 6))
corr = df[['PM2.5', 'PM10', 'NO2', 'O3', 'AQI']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap - Pollutants vs AQI")
plt.show()


"""🔍 Explanation of Each Plot

Line Plot – Shows how pollutants like PM2.5 change over time (useful for trend analysis).

Scatter Plot – Helps see correlation (e.g., higher PM2.5 usually means higher AQI).

Bar Chart – Compares average pollutant levels to see which one dominates.

Histogram – Distribution of AQI (whether most days are Good, Moderate, or Poor).

Boxplot – Detects outliers (spikes in pollutants on certain days).

Subplots – Compare PM2.5 and AQI together (side by side).

Heatmap – Statistical correlation between pollutants and AQI."""
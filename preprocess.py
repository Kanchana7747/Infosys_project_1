import pandas as pd
import numpy as np
import os

# ------------------------
# Load raw data
# ------------------------
def load_raw(path):
    """Load CSV and lowercase column names"""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# ------------------------
# Ensure timestamp is index
# ------------------------
def ensure_index_time(df, time_col='timestamp'):
    """Set a datetime index, detect timestamp column if missing"""
    if time_col not in df.columns:
        for c in df.columns:
            if any(x in c.lower() for x in ['date', 'time', 'datetime', 'timestamp']):
                time_col = c
                break
        else:
            raise KeyError("No timestamp-like column found in dataframe")
    
    df = df.dropna(subset=[time_col]).copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    df.index.name = 'timestamp'  # ensure consistent name
    return df

# ------------------------
# Resample station data
# ------------------------
def resample_station(df, station_id, freq='h', fill_method='interpolate'):
    s = df[df['station_id'] == station_id].copy()

    numeric_cols = s.select_dtypes(include=np.number).columns
    non_numeric_cols = s.select_dtypes(exclude=np.number).columns

    # Resample numeric and non-numeric separately
    s_numeric = s[numeric_cols].resample(freq).mean()
    s_non_numeric = s[non_numeric_cols].resample(freq).first()

    # Combine
    s = pd.concat([s_numeric, s_non_numeric], axis=1)

    # Fill missing values
    if fill_method == 'interpolate':
        s[numeric_cols] = s[numeric_cols].interpolate(limit_direction='both', method='linear')
        s[non_numeric_cols] = s[non_numeric_cols].ffill()
    elif fill_method == 'ffill':
        s = s.ffill()

    return s

# ------------------------
# Outlier detection
# ------------------------
def detect_iqr_outliers(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (series < lower) | (series > upper)
    return mask

def cap_outliers(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return series.clip(lower, upper)

# ------------------------
# Feature engineering
# ------------------------
def add_time_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def add_lag_features(df, cols, lags=[1, 3, 24, 168]):
    for c in cols:
        for lag in lags:
            df[f'{c}_lag_{lag}'] = df[c].shift(lag)
    return df

def add_rolling(df, cols, windows=[3, 24]):
    for c in cols:
        for w in windows:
            df[f'{c}_roll_{w}'] = df[c].rolling(window=w, min_periods=1).mean()
    return df

# ------------------------
# Pipeline
# ------------------------
def pipeline(raw_path, outdir='data_processed', freq='h'):
    os.makedirs(outdir, exist_ok=True)

    raw = load_raw(raw_path)
    raw = ensure_index_time(raw)

    # Add a default station_id if missing
    if 'station_id' not in raw.columns:
        raw['station_id'] = 'Central'

    stations = raw['station_id'].dropna().unique()
    frames = []

    for st in stations:
        s = resample_station(raw, st, freq=freq, fill_method='interpolate')

        # Outlier capping for pollutants
        pollutant_cols = [c for c in s.columns if c.lower() in ['pm25', 'pm10', 'no2', 'o3', 'co']]
        for p in pollutant_cols:
            if p in s.columns:
                s[p] = cap_outliers(s[p])

        # Add features
        s = add_time_features(s)
        s = add_lag_features(s, pollutant_cols)
        s = add_rolling(s, pollutant_cols)
        s['station_id'] = st

        frames.append(s)

    all_df = pd.concat(frames).sort_index()
    all_df.index.name = 'timestamp'  # ensure index name

    # Save hourly data
    hourly_path = os.path.join(outdir, f'hourly_{freq}.parquet')
    all_df.to_parquet(hourly_path)

    # Daily aggregation (numeric only)
    daily = all_df.copy()
    numeric_cols = daily.select_dtypes(include=np.number).columns
    daily = daily.groupby('station_id')[numeric_cols].resample('D').mean()
    daily.reset_index(level=0, drop=False, inplace=True)  # keep station_id as column
    daily_path = os.path.join(outdir, 'daily.parquet')
    daily.to_parquet(daily_path)

    print("Saved", hourly_path, daily_path)
    return hourly_path, daily_path

# ------------------------
# Main entry point
# ------------------------
if __name__ == "__main__":
    import sys
    raw_path = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\slaxm\OneDrive\Documents\Air_Aware\city_hour.csv'
    pipeline(raw_path)

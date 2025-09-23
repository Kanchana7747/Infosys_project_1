import pandas as pd
import os
from datetime import datetime

# ------------------------
# Load CSV
# ------------------------
def load_csv(filepath=r'C:\Users\slaxm\OneDrive\Documents\city_hour.csv'):
    """Load a CSV file into a DataFrame"""
    df = pd.read_csv(filepath)
    return df

# ------------------------
# Standardize column names
# ------------------------
def standardize_columns(df):
    """Lowercase column names and map to standard names"""
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    
    mapping = {}
    for col in df.columns:
        if 'pm2' in col:
            mapping[col] = 'pm25'
        elif 'pm10' in col:
            mapping[col] = 'pm10'
        elif 'no2' in col:
            mapping[col] = 'no2'
        elif 'o3' in col:
            mapping[col] = 'o3'
        elif 'co' == col:
            mapping[col] = 'co'
        elif 'date' in col or 'timestamp' in col or 'datetime' in col:
            mapping[col] = 'timestamp'
        elif 'city' in col:
            mapping[col] = 'city'
        elif 'station' in col or 'site' in col:
            mapping[col] = 'station_id'
    
    df = df.rename(columns=mapping)
    return df

# ------------------------
# Parse timestamp
# ------------------------
def parse_timestamp(df, tz=None):
    """Convert timestamp column to datetime, optionally localize timezone"""
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if tz:
        df['timestamp'] = df['timestamp'].dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
    return df

# ------------------------
# Save raw DataFrame
# ------------------------
def save_raw(df, outdir='data_raw', name=None):
    """Save DataFrame as parquet"""
    os.makedirs(outdir, exist_ok=True)
    if name is None:
        name = datetime.utcnow().strftime('raw_%Y%m%d_%H%M.parquet')
    outpath = os.path.join(outdir, name)
    df.to_parquet(outpath, index=False)
    print("Saved raw to", outpath)
    return outpath

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    df = load_csv(r"C:\Users\slaxm\OneDrive\Documents\air_quality_milestone1.csv")  # Load the milestone1 CSV
    df = standardize_columns(df)                 # Standardize column names
    df = parse_timestamp(df)                     # Parse timestamp
    save_raw(df)                                 # Save to parquet

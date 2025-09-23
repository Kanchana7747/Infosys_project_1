import mysql.connector
from mysql.connector import Error

# ------------------------
# Database and tables setup
# ------------------------
DB_NAME = "airq"

TABLES = {
    "stations": """
        CREATE TABLE IF NOT EXISTS stations (
            station_id VARCHAR(64) PRIMARY KEY,
            city VARCHAR(100),
            latitude DECIMAL(9,6),
            longitude DECIMAL(9,6),
            metadata JSON
        );
    """,
    "measurements_hourly": """
        CREATE TABLE IF NOT EXISTS measurements_hourly (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            station_id VARCHAR(64),
            ts DATETIME,
            pm25 FLOAT,
            pm10 FLOAT,
            no2 FLOAT,
            o3 FLOAT,
            co FLOAT,
            temperature FLOAT,
            humidity FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES stations(station_id)
        );
    """,
    "features_hourly": """
        CREATE TABLE IF NOT EXISTS features_hourly (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            station_id VARCHAR(64),
            ts DATETIME,
            feature_name VARCHAR(100),
            feature_value FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "models": """
        CREATE TABLE IF NOT EXISTS models (
            model_id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(128),
            version VARCHAR(32),
            path VARCHAR(255),
            metrics JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """,
    "alerts": """
        CREATE TABLE IF NOT EXISTS alerts (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            station_id VARCHAR(64),
            ts DATETIME,
            pollutant VARCHAR(32),
            predicted_value FLOAT,
            aqi_category VARCHAR(50),
            message TEXT,
            notified BOOLEAN DEFAULT FALSE
        );
    """
}

# ------------------------
# Connect to MySQL
# ------------------------
def connect_to_mysql(create_db=False):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root'
        )
        if connection.is_connected():
            print("Connected to MySQL server")
            cursor = connection.cursor()
            if create_db:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
                print(f"Database '{DB_NAME}' ensured")
            cursor.close()
            return connection
    except Error as e:
        print("Error while connecting to MySQL:", e)
        return None

# ------------------------
# Create tables
# ------------------------
def create_tables(connection):
    try:
        connection.database = DB_NAME
        cursor = connection.cursor()
        for table_name, ddl in TABLES.items():
            cursor.execute(ddl)
            print(f"Table '{table_name}' ensured")
        cursor.close()
    except Error as e:
        print("Error creating tables:", e)

# ------------------------
# Fetch stations
# ------------------------
def fetch_stations(connection):
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM stations")
        results = cursor.fetchall()
        if results:
            for row in results:
                print(row)
        else:
            print("No stations found")
        cursor.close()
    except Error as e:
        print("Error fetching stations:", e)

# ------------------------
# Main
# ------------------------
def main():
    conn = connect_to_mysql(create_db=True)
    if conn:
        create_tables(conn)
        fetch_stations(conn)
        conn.close()
        print("Connection closed")

if __name__ == "__main__":
    main()

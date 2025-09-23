import mysql.connector
from mysql.connector import Error

def connect_to_mysql():
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host='localhost',       # or your MySQL server IP
            user='your_username',   # replace with your MySQL username
            password='your_password', # replace with your MySQL password
            database='airq'         # your database name
        )

        if connection.is_connected():
            print("Connected to MySQL database")
            return connection

    except Error as e:
        print("Error while connecting to MySQL:", e)
        return None

def fetch_stations(connection):
    """Example query: fetch all stations"""
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM stations")
    results = cursor.fetchall()
    for row in results:
        print(row)
    cursor.close()

def main():
    conn = connect_to_mysql()
    if conn:
        fetch_stations(conn)
        conn.close()
        print("Connection closed")

if __name__ == "__main__":
    main()

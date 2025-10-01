import streamlit as st
import pandas as pd
import mysql.connector
from hashlib import sha256
import requests

# ----------------------------
# MySQL connection
# ----------------------------
def create_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            database='air_qualities',
            user='root',
            password='root'
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {e}")
        return None

# ----------------------------
# Verify login credentials
# ----------------------------
def verify_user(username, password):
    conn = create_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user:
            hashed_password = sha256(password.encode()).hexdigest()
            if hashed_password == user['password']:
                return user['role']
    return None

# ----------------------------
# Streamlit Login UI
# ----------------------------
if 'role' not in st.session_state:
    st.title("🔒 Air Quality Dashboard Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        role = verify_user(username, password)
        if role:
            st.session_state['role'] = role
            st.session_state['username'] = username
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

# ----------------------------
# Main App After Login
# ----------------------------
if 'role' in st.session_state:
    role = st.session_state['role']
    st.header(f"Welcome, {st.session_state['username']} ({role})")

    # ----------------------------
    # Sidebar Navbar
    # ----------------------------
    options = []
    if role == 'admin':
        options += ["Admin Panel", "Pollutant Description", "Live Air Quality Data"]
    else:
        options += ["Pollutant Description", "Live Air Quality Data"]

    page = st.sidebar.radio("Navigate", options)

    # ----------------------------
    # Admin Panel
    # ----------------------------
    if page == "Admin Panel" and role == 'admin':
        st.subheader("📁 Upload Air Quality CSV")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.info("Upload a CSV to view data.")

        st.subheader("⚙️ Admin Panel: Create New User")
        new_username = st.text_input("New Username", key="new_user")
        new_password = st.text_input("New Password", type="password", key="new_pass")
        new_role = st.selectbox("Role", ["user"])

        if st.button("Create User"):
            if not new_username or not new_password:
                st.warning("Username and password cannot be empty.")
            else:
                hashed_pass = sha256(new_password.encode()).hexdigest()
                conn = create_connection()
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                            (new_username, hashed_pass, new_role)
                        )
                        conn.commit()
                        cursor.close()
                        st.success(f"User '{new_username}' created successfully!")
                    except mysql.connector.IntegrityError:
                        st.error("Username already exists!")
                    finally:
                        conn.close()

        st.subheader("Existing Users")
        conn = create_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, username, role FROM users")
            users = cursor.fetchall()
            st.table(users)
            cursor.close()
            conn.close()

    # ----------------------------
    # Pollutant Description
    # ----------------------------
    elif page == "Pollutant Description":
        st.subheader("🧪 Pollutant Description")
        pollutants_info = {
            "PM2.5": "Fine particulate matter ≤2.5 µm, can penetrate lungs.",
            "PM10": "Particulate matter ≤10 µm, affects respiratory system.",
            "NO2": "Nitrogen dioxide, mainly from vehicles, causes respiratory issues.",
            "O3": "Ozone, can irritate lungs and worsen asthma.",
            "CO": "Carbon monoxide, reduces oxygen delivery in the body.",
            "SO2": "Sulfur dioxide, causes respiratory problems."
        }
        for pol, desc in pollutants_info.items():
            st.markdown(f"**{pol}**: {desc}")


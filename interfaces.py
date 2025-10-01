import streamlit as st
import pandas as pd
import mysql.connector
from hashlib import sha256
import os

# ---------------------------- MYSQL CONNECTION ----------------------------
def get_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='root',
        database='app_db'
    )

# ---------------------------- PASSWORD HASH ----------------------------
def hash_password(password):
    return sha256(password.encode()).hexdigest()

# ---------------------------- USER REGISTRATION ----------------------------
def register_user(username, password, role='user'):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                       (username, hash_password(password), role))
        conn.commit()
        st.success("User registered successfully!")
    except mysql.connector.errors.IntegrityError:
        st.error("Username already exists.")
    finally:
        cursor.close()
        conn.close()

# ---------------------------- LOGIN ----------------------------
def login(username, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s",
                   (username, hash_password(password)))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

# ---------------------------- UPLOAD FILES (ADMIN) ----------------------------
def upload_file(file, uploaded_by):
    if file is not None:
        filename = file.name
        save_path = f"./uploads/{filename}"
        os.makedirs("./uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        # Save to DB
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO files (filename, filepath, uploaded_by) VALUES (%s, %s, %s)",
                       (filename, save_path, uploaded_by))
        conn.commit()
        cursor.close()
        conn.close()
        st.success(f"{filename} uploaded successfully!")

# ---------------------------- STREAMLIT APP ----------------------------
st.title("MySQL + Python Dashboard App")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.subheader("Create New Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    role = st.selectbox("Role", ["user", "admin"])
    if st.button("Register"):
        register_user(username, password, role)

elif choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        user = login(username, password)
        if user:
            st.success(f"Logged in as {user['username']} ({user['role']})")
            
            # ---------------------------- NAVBAR ----------------------------
            nav_options = ["Dashboard"]
            if user['role'] == 'admin':
                nav_options.append("Admin Panel")
            
            page = st.sidebar.radio("Navigation", nav_options)
            
            if page == "Dashboard":
                st.subheader("User Dashboard")
                # Example: Display uploaded files
                conn = get_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM files")
                files = cursor.fetchall()
                cursor.close()
                conn.close()
                if files:
                    df = pd.DataFrame(files)
                    st.dataframe(df)
                else:
                    st.info("No data available.")
            
            elif page == "Admin Panel" and user['role'] == 'admin':
                st.subheader("Admin Panel - Upload Files")
                file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])
                if st.button("Upload"):
                    upload_file(file, user['id'])
        else:
            st.error("Invalid username or password")

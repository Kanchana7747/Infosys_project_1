import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="City-wise Data Viewer", layout="wide")
st.title("City-wise Data Viewer")

# File uploader
uploaded_file = st.file_uploader(r"C:\Users\slaxm\OneDrive\Documents\air_quality_milestone1.csv", type=["csv"])

if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success("File uploaded successfully!")

    # Show the full dataset
    st.subheader("Full Dataset")
    st.dataframe(df)

    # City filter
    if "station_id" in df.columns:
        city_list = df["station_id"].dropna().unique()
        selected_city = st.selectbox("Select a City", city_list)

        # Filter data for the selected city
        city_data = df[df["station_id"] == selected_city]

        st.subheader(f"Data for {selected_city}")
        st.dataframe(city_data)

        # Date column check
        date_col = None
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
                break

        if date_col:
            city_data[date_col] = pd.to_datetime(city_data[date_col], errors='coerce')
            city_data = city_data.dropna(subset=[date_col])
            city_data = city_data.sort_values(by=date_col)

            # Select numeric column for plotting
            numeric_cols = city_data.select_dtypes(include=['float64','int64']).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select a column to plot", numeric_cols)

                # Line chart
                st.subheader(f"Line Chart for {selected_col} in {selected_city}")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(city_data[date_col], city_data[selected_col], marker='o', linestyle='-')
                ax.set_xlabel("Date")
                ax.set_ylabel(selected_col)
                ax.set_title(f"{selected_col} over time in {selected_city}")
                st.pyplot(fig)
            else:
                st.warning("No numeric columns available to plot.")
        else:
            st.warning("No date column found in the dataset. Please include a 'date' column.")
    else:
        st.warning("The dataset must have a 'city' column for filtering.")
else:
    st.info("Please upload a CSV file to begin.")

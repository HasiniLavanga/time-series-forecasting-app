import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting App",
    layout="wide"
)

# Title
st.title("📈 Time Series Forecasting App")

st.write("Upload a CSV file to visualize time series data.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

# If file uploaded
if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    # Show dataset
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # Column selection
    columns = df.columns.tolist()

    date_col = st.selectbox(
        "Select Date Column",
        columns
    )

    target_col = st.selectbox(
        "Select Target Column",
        columns
    )

    # Convert date column
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date
    df = df.sort_values(by=date_col)

    # Dataset information
    st.subheader("📊 Dataset Information")
    st.write(df.describe())

    # Line chart
    st.subheader("📉 Time Series Visualization")

    fig = px.line(
        df,
        x=date_col,
        y=target_col,
        title=f"{target_col} Over Time"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.info("Please upload a CSV file to continue.")
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# Page config
st.set_page_config(
    page_title="Time Series Forecasting App",
    layout="wide"
)

# Title
st.title("📈 Time Series Forecasting App")

st.write("Upload a CSV file for forecasting.")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    # Preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # Select columns
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

    # Sort values
    df = df.sort_values(by=date_col)

    # Visualization
    st.subheader("📉 Time Series Visualization")

    fig = px.line(
        df,
        x=date_col,
        y=target_col,
        title=f"{target_col} Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast days
    forecast_days = st.slider(
        "Select Forecast Days",
        min_value=7,
        max_value=365,
        value=30
    )

    # Prepare data for Prophet
    prophet_df = df[[date_col, target_col]]

    prophet_df.columns = ["ds", "y"]

    # Train model
    st.subheader("🤖 Training Prophet Model...")

    model = Prophet()

    model.fit(prophet_df)

    # Future dataframe
    future = model.make_future_dataframe(
        periods=forecast_days
    )

    # Forecast
    forecast = model.predict(future)

    # Forecast preview
    st.subheader("📊 Forecast Results")

    st.dataframe(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
    )

    # Forecast chart
    st.subheader("📈 Forecast Visualization")

    forecast_fig = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="Forecast Prediction"
    )

    # Add confidence intervals
    forecast_fig.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode='lines',
        name='Upper Bound'
    )

    forecast_fig.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode='lines',
        name='Lower Bound'
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

else:
    st.info("Please upload a CSV file.")

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting App",
    layout="wide"
)

# Title
st.title("📈 Time Series Forecasting App")

st.write("Upload a CSV file for forecasting and model comparison.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV file
    df = pd.read_csv(uploaded_file)

    # Dataset preview
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

    # Sort values by date
    df = df.sort_values(by=date_col)

    # Historical chart
    st.subheader("📉 Historical Data")

    historical_fig = px.line(
        df,
        x=date_col,
        y=target_col,
        title=f"{target_col} Over Time"
    )

    st.plotly_chart(
        historical_fig,
        use_container_width=True
    )

    # Forecast days slider
    forecast_days = st.slider(
        "Select Forecast Days",
        min_value=7,
        max_value=365,
        value=30
    )

    # =========================================
    # PROPHET MODEL
    # =========================================

    st.subheader("🤖 Prophet Forecast")

    # Prepare data for Prophet
    prophet_df = df[[date_col, target_col]].copy()

    prophet_df.columns = ["ds", "y"]

    # Convert target column to numeric
    prophet_df["y"] = pd.to_numeric(
        prophet_df["y"],
        errors="coerce"
    )

    # Remove missing values
    prophet_df = prophet_df.dropna()

    # Train Prophet model
    prophet_model = Prophet()

    prophet_model.fit(prophet_df)

    # Create future dataframe
    future = prophet_model.make_future_dataframe(
        periods=forecast_days
    )

    # Generate forecast
    prophet_forecast = prophet_model.predict(future)

    # Forecast table
    st.subheader("📊 Prophet Forecast Results")

    st.dataframe(
        prophet_forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].tail()
    )

    # Prophet chart
    prophet_fig = px.line(
        prophet_forecast,
        x="ds",
        y="yhat",
        title="Prophet Forecast"
    )

    # Add upper confidence interval
    prophet_fig.add_scatter(
        x=prophet_forecast["ds"],
        y=prophet_forecast["yhat_upper"],
        mode="lines",
        name="Upper Bound"
    )

    # Add lower confidence interval
    prophet_fig.add_scatter(
        x=prophet_forecast["ds"],
        y=prophet_forecast["yhat_lower"],
        mode="lines",
        name="Lower Bound"
    )

    st.plotly_chart(
        prophet_fig,
        use_container_width=True
    )

    # =========================================
    # ARIMA MODEL
    # =========================================

    st.subheader("📈 ARIMA Forecast")

    # Convert target column to numeric
    series = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    # Remove missing values
    series = series.dropna()

    # Train ARIMA model
    arima_model = ARIMA(
        series,
        order=(5, 1, 0)
    )

    arima_model_fit = arima_model.fit()

    # Generate ARIMA forecast
    arima_forecast = arima_model_fit.forecast(
        steps=forecast_days
    )

    # Create future dates
    future_dates = pd.date_range(
        start=df[date_col].max(),
        periods=forecast_days + 1,
        freq="D"
    )[1:]

    # Forecast dataframe
    arima_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": arima_forecast
    })

    # ARIMA chart
    arima_fig = px.line(
        arima_df,
        x="Date",
        y="Forecast",
        title="ARIMA Forecast"
    )

    st.plotly_chart(
        arima_fig,
        use_container_width=True
    )

    # =========================================
    # MODEL EVALUATION
    # =========================================

    st.subheader("📊 Model Evaluation Metrics")

    # Train-test split
    train_size = int(len(series) * 0.8)

    train = series[:train_size]

    test = series[train_size:]

    # Train evaluation model
    eval_model = ARIMA(
        train,
        order=(5, 1, 0)
    )

    eval_model_fit = eval_model.fit()

    # Predictions
    predictions = eval_model_fit.forecast(
        steps=len(test)
    )

    # Calculate metrics
    mae = mean_absolute_error(
        test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            predictions
        )
    )

    # Metrics dataframe
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE"],
        "Value": [mae, rmse]
    })

    st.table(metrics_df)

else:
    st.info("Please upload a CSV file.")

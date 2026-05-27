import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Time Series Forecasting App",
    layout="wide"
)

# =========================================
# TITLE
# =========================================

st.title("📈 Time Series Forecasting App")

st.write(
    "Multi-model forecasting using Prophet, ARIMA, and XGBoost."
)

# =========================================
# FILE UPLOADER
# =========================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =========================================
# MAIN APPLICATION
# =========================================

if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    # =========================================
    # DATASET PREVIEW
    # =========================================

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # =========================================
    # COLUMN SELECTION
    # =========================================

    columns = df.columns.tolist()

    date_col = st.selectbox(
        "Select Date Column",
        columns
    )

    target_col = st.selectbox(
        "Select Target Column",
        columns
    )

    # =========================================
    # DATA CLEANING
    # =========================================

    # Convert date column
    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    # Convert target column to numeric
    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna()

    # Sort by date
    df = df.sort_values(by=date_col)

    # =========================================
    # HISTORICAL VISUALIZATION
    # =========================================

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

    # =========================================
    # FORECAST DAYS
    # =========================================

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

    prophet_df = df[[date_col, target_col]].copy()

    prophet_df.columns = ["ds", "y"]

    # Ensure correct datetime format
    prophet_df["ds"] = pd.to_datetime(
        prophet_df["ds"],
        errors="coerce"
    )

    # Ensure numeric target
    prophet_df["y"] = pd.to_numeric(
        prophet_df["y"],
        errors="coerce"
    )

    # Remove invalid rows
    prophet_df = prophet_df.dropna()

    # Train Prophet model
    prophet_model = Prophet()

    prophet_model.fit(prophet_df)

    # Create future dataframe
    future = prophet_model.make_future_dataframe(
        periods=forecast_days,
        freq="D"
    )

    # Generate forecast
    prophet_forecast = prophet_model.predict(future)

    # Forecast results table
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

    # Confidence intervals
    prophet_fig.add_scatter(
        x=prophet_forecast["ds"],
        y=prophet_forecast["yhat_upper"],
        mode="lines",
        name="Upper Bound"
    )

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

    series = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    series = series.dropna()

    # Train ARIMA
    arima_model = ARIMA(
        series,
        order=(5, 1, 0)
    )

    arima_model_fit = arima_model.fit()

    # Forecast
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
    # XGBOOST MODEL
    # =========================================

    st.subheader("⚡ XGBoost Forecast")

    # Create separate dataframe
    xgb_df = pd.DataFrame()

    # Add target column
    xgb_df["target"] = df[target_col]

    # Create lag features
    xgb_df["lag_1"] = xgb_df["target"].shift(1)

    xgb_df["lag_2"] = xgb_df["target"].shift(2)

    xgb_df["lag_3"] = xgb_df["target"].shift(3)

    # Remove missing rows
    xgb_df = xgb_df.dropna()

    # Features and target
    X = xgb_df[["lag_1", "lag_2", "lag_3"]]

    y = xgb_df["target"]

    # Train-test split
    split_size = int(len(X) * 0.8)

    X_train = X[:split_size]

    X_test = X[split_size:]

    y_train = y[:split_size]

    y_test = y[split_size:]

    # Train model
    xgb_model = XGBRegressor()

    xgb_model.fit(X_train, y_train)

    # Predictions
    xgb_predictions = xgb_model.predict(X_test)

    # Metrics
    xgb_mae = mean_absolute_error(
        y_test,
        xgb_predictions
    )

    xgb_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            xgb_predictions
        )
    )

    # Results dataframe
    xgb_results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": xgb_predictions
    })

    # XGBoost chart
    xgb_fig = px.line(
        xgb_results,
        title="XGBoost Predictions"
    )

    st.plotly_chart(
        xgb_fig,
        use_container_width=True
    )

    # =========================================
    # MODEL COMPARISON
    # =========================================

    st.subheader("📊 Model Comparison Metrics")

    # Train-test split for ARIMA
    train_size = int(len(series) * 0.8)

    train = series[:train_size]

    test = series[train_size:]

    # ARIMA evaluation
    eval_model = ARIMA(
        train,
        order=(5, 1, 0)
    )

    eval_model_fit = eval_model.fit()

    arima_predictions = eval_model_fit.forecast(
        steps=len(test)
    )

    # ARIMA metrics
    arima_mae = mean_absolute_error(
        test,
        arima_predictions
    )

    arima_rmse = np.sqrt(
        mean_squared_error(
            test,
            arima_predictions
        )
    )

    # Metrics table
    metrics_df = pd.DataFrame({
        "Model": ["ARIMA", "XGBoost"],
        "MAE": [arima_mae, xgb_mae],
        "RMSE": [arima_rmse, xgb_rmse]
    })

    st.table(metrics_df)

else:
    st.info("Please upload a CSV file.")

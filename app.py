import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("smart_grid_dataset.csv", parse_dates=["Timestamp"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Timestamp", "Power Consumption (kW)"])
    return df

# ---- Feature Engineering ----
def add_features(df):
    df = df.sort_values("Timestamp")
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek
    df["rolling_mean_24h"] = df["Power Consumption (kW)"].rolling(window=4, min_periods=1).mean()
    return df

# ---- Modeling ----
def train_model(df):
    features = ["hour", "day_of_week", "rolling_mean_24h"]
    X = df[features]
    y = df["Power Consumption (kW)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# ---- App Layout ----
st.title("🔌 Smart Grid Load Forecasting Dashboard")

# Load and process data
df = load_data()
df = add_features(df)

# Sidebar
st.sidebar.header("Configuration")
show_raw = st.sidebar.checkbox("Show Raw Data")
show_forecast = st.sidebar.checkbox("Show Forecast", value=True)
show_shap = st.sidebar.checkbox("Show SHAP Explanation", value=True)

# Display raw data
if show_raw:
    st.subheader("📊 Raw Smart Grid Data")
    st.dataframe(df.head(50))

# Time Series Visualization
st.subheader("⏱ Power Consumption Over Time")
st.line_chart(df.set_index("Timestamp")["Power Consumption (kW)"])

# Forecasting
if show_forecast:
    st.subheader("🔮 Forecasting with Random Forest")
    model, X_test, y_test = train_model(df)
    y_pred = model.predict(X_test)
    forecast_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    }, index=y_test.index)
    st.line_chart(forecast_df)

# SHAP Explainability
if show_shap:
    st.subheader("🧠 Feature Importance (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("your_data.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    return df

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# Feature engineering
def add_features(df, target_column="load"):
    df[f"{target_column}_rolling_mean_24h"] = df[target_column].rolling(window=24, min_periods=1).mean()
    df[f"{target_column}_rolling_std_24h"] = df[target_column].rolling(window=24, min_periods=1).std()
    df = df.fillna(method="bfill")  # Ensure no NaNs for model
    return df

# SHAP plotting (modularized to avoid ScriptRunContext errors)
from shap_summary_plot import plot_shap_summary

# Streamlit App
st.set_page_config(page_title="Smart Grid Forecast", layout="wide")
st.title("🔌 Smart Grid Load Forecasting Dashboard")

# Load
df = load_data()
model = load_model()

df = add_features(df)
features = df.drop(columns=["timestamp", "load"])

# Prediction
predictions = model.predict(features)
df["prediction"] = predictions

# Plot forecast
st.subheader("📊 Load Forecast vs Actual")
st.line_chart(df[["load", "prediction"]])

# SHAP Explainability
st.subheader("🤖 SHAP Feature Importance")
plot_shap_summary(model, features)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Smart Grid Load Forecasting Dashboard", layout="wide")

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("smart_grid_dataset.csv")  # Remove parse_dates
    df['timestamp'] = pd.to_datetime(df['date_time'])  # Replace 'date_time' with actual column name
    df = df.dropna()
    return df


# ---- Feature Engineering ----
def add_features(df):
    target_column = "load"  # Ensure this column exists in your dataset
    if target_column not in df.columns:
        st.error(f"❌ Column '{target_column}' not found in the dataset. Available columns: {df.columns.tolist()}")
        st.stop()
    df = df.copy()
    df.set_index("timestamp", inplace=True)
    df[f"{target_column}_rolling_mean_24h"] = df[target_column].rolling(window=24, min_periods=1).mean()
    df[f"{target_column}_rolling_std_24h"] = df[target_column].rolling(window=24, min_periods=1).std()
    df = df.dropna()
    return df.reset_index()

# ---- Load Model ----
def load_model():
    try:
        model = joblib.load("model.joblib")
        return model
    except:
        st.warning("Model file not found. Training a new model...")
        return None

# ---- Main App ----
st.title("🔌 Smart Grid Load Forecasting Dashboard")

# Load data
df = load_data()
df = add_features(df)

# Sidebar
st.sidebar.header("Options")
show_raw = st.sidebar.checkbox("Show raw data", False)

if show_raw:
    st.subheader("Raw Data")
    st.write(df.head())

# Train/Test Split
feature_cols = [col for col in df.columns if col not in ["timestamp", "load"]]
X = df[feature_cols]
y = df["load"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or train model
model = load_model()
if model is None:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "model.joblib")

# Predict
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Display Results
st.subheader("📊 Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.set_xlabel("Actual Load")
ax.set_ylabel("Predicted Load")
ax.set_title("Actual vs Predicted Load")
st.pyplot(fig)

# SHAP Explainability
st.subheader("🔍 Feature Importance (SHAP)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

fig_shap, ax_shap = plt.subplots()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
st.pyplot(fig_shap)

st.success("✅ App loaded successfully!")

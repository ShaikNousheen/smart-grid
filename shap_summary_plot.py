import streamlit as st
import shap
import matplotlib.pyplot as plt

def plot_shap_summary(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig)

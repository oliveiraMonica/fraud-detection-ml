import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# -----------------------------
# Load trained model
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

model = joblib.load(MODEL_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("💳 Fraud Detection System")

st.write(
    "This application uses a machine learning model "
    "to predict whether a credit card transaction is fraudulent."
)

st.subheader("Enter transaction features")

# Example: simplified input
amount = st.number_input("Transaction Amount", value=100.0)

v1 = st.number_input("Feature V1", value=0.0)
v2 = st.number_input("Feature V2", value=0.0)
v3 = st.number_input("Feature V3", value=0.0)

# For demonstration we create a dummy vector
features = [0]*30
features[0] = v1
features[1] = v2
features[2] = v3
features[-1] = amount

if st.button("Predict Fraud"):

    features_array = np.array(features).reshape(1, -1)

    prediction = model.predict(features_array)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent transaction detected!")
    else:
        st.success("✅ Transaction appears legitimate")
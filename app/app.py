import streamlit as st
import joblib
import pandas as pd
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details below and click **Predict**.")

# -----------------------------
# Load Model & Scaler
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "heart_disease_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

sex_male = 1 if sex == "Male" else 0

cigsPerDay = st.number_input(
    "Cigarettes Smoked Per Day",
    min_value=0,
    max_value=60,
    value=0
)

totChol = st.number_input(
    "Total Cholesterol",
    min_value=100,
    max_value=700,
    value=200
)

sysBP = st.number_input(
    "Systolic Blood Pressure",
    min_value=80,
    max_value=250,
    value=120
)

glucose = st.number_input(
    "Glucose Level",
    min_value=40,
    max_value=400,
    value=80
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):

    sample = pd.DataFrame(
        [[
            age,
            sex_male,
            cigsPerDay,
            totChol,
            sysBP,
            glucose
        ]],
        columns=[
            "age",
            "Sex_male",
            "cigsPerDay",
            "totChol",
            "sysBP",
            "glucose"
        ]
    )

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
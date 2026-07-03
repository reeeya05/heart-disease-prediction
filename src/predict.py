import joblib
import numpy as np
import pandas as pd
import os

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Load model + scaler
model = joblib.load(os.path.join(BASE_DIR, "models", "heart_disease_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# Example input (CHANGE values as per your dataset features)
sample_input = pd.DataFrame([[ 55, 1, 130, 250, 0, 1]], columns=[ "age","Sex_male","cigsPerDay","totChol","sysBP","glucose"])

# Scale input
sample_input = scaler.transform(sample_input)

# Predict
prediction = model.predict(sample_input)

if prediction[0] == 1:
    print("⚠️ Heart Disease Risk")
else:
    print("✅ No Risk")
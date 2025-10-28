import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# -------------------------------
# Load model and preprocessing files
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("churn_model.h5")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

st.set_page_config(page_title="Bank Customer Churn Prediction", layout="centered")

st.title("🏦 Bank Customer Churn Prediction App")
st.write("Enter customer details below to predict whether the customer will churn (exit) or not.")

# -------------------------------
# Input fields
# -------------------------------
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_credit_card = st.selectbox("Has Credit Card?", ["No", "Yes"])
is_active_member = st.selectbox("Is Active Member?", ["No", "Yes"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0)
gender = st.selectbox("Gender", ["Female", "Male"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# -------------------------------
# Preprocessing
# -------------------------------
if st.button("🔍 Predict Churn"):
    # Encode gender
    gender_encoded = label_encoder.transform([gender])[0]  # 0 for Female, 1 for Male

    # One-hot encode Geography (France, Germany, Spain)
    geography_france = 1 if geography == "France" else 0
    geography_germany = 1 if geography == "Germany" else 0
    geography_spain = 1 if geography == "Spain" else 0

    # Convert categorical to numeric
    has_credit_card = 1 if has_credit_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0

    # Log transform Age (same as training)
    age_log = np.log1p(age)

    # Create input dataframe (same column order as training)
    data = np.array([[credit_score, gender_encoded, age_log, tenure, balance, num_of_products,
                      has_credit_card, is_active_member, estimated_salary,
                      geography_france, geography_germany, geography_spain]])

    # Scale input
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)
    result = (prediction > 0.5).astype(int)[0][0]

    # -------------------------------
    # Display result
    # -------------------------------
    if result == 1:
        st.error("🚨 The customer is **likely to churn (exit)**.")
    else:
        st.success("✅ The customer is **likely to stay with the bank**.")

    st.write(f"**Predicted Probability of Churn:** {float(prediction[0][0]):.2f}")


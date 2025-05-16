import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("../notebooks/heart_disease_model.pkl", "rb"))

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Enter your details below:")

# User Inputs
age = st.number_input("Age", min_value=20, max_value=100, value=40)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)

# Predict Button
if st.button("Predict"):
    features = np.array([age, chol, bp]).reshape(1, -1)
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("High Risk: You may have heart disease. Consult a doctor.")
    else:
        st.success("Low Risk: No heart disease detected.")


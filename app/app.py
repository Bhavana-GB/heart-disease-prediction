import streamlit as st
import pandas as pd
import pickle

# Load saved pipeline
model_pipeline = pickle.load(open("../notebooks/heart_disease_pipeline.pkl", "rb"))

st.title("Heart Disease Prediction App")
st.write("Enter your health details:")


age = st.number_input('Age', min_value=20, max_value=100, value=40)

# sex: 1 = male, 0 = female 
sex = st.selectbox('Sex', options=['Male', 'Female'])
sex_val = 1 if sex == 'Male' else 0

# cp (chest pain type)
cp = st.selectbox('Chest Pain Type',
                  options=[1, 2, 3, 4],
                  format_func=lambda x: {
                      1: 'Typical Angina',
                      2: 'Atypical Angina',
                      3: 'Non-anginal Pain',
                      4: 'Asymptomatic'
                  }[x])

trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)

chol = st.number_input('Cholesterol Level', min_value=100, max_value=400, value=200)

# fbs (fasting blood sugar > 120 mg/dl): 1 = true, 0 = false
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', options=['Yes', 'No'])
fbs_val = 1 if fbs == 'Yes' else 0

# restecg (resting ECG results)
restecg = st.selectbox('Resting ECG Results',
                       options=[0, 1, 2],
                       format_func=lambda x: {
                           0: 'Normal',
                           1: 'Having ST-T wave abnormality',
                           2: 'Showing probable or definite left ventricular hypertrophy'
                       }[x])

thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)

# exang (exercise induced angina): 1 = yes, 0 = no
exang = st.selectbox('Exercise Induced Angina?', options=['Yes', 'No'])
exang_val = 1 if exang == 'Yes' else 0

oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, step=0.1, value=1.0)

# slope (slope of the peak exercise ST segment)
slope = st.selectbox('Slope of the Peak Exercise ST Segment',
                     options=[1, 2, 3],
                     format_func=lambda x: {
                         1: 'Upsloping',
                         2: 'Flat',
                         3: 'Downsloping'
                     }[x])

# ca (number of major vessels colored by fluoroscopy)
ca = st.selectbox('Number of Major Vessels (0-3)', options=[0, 1, 2, 3])


thal = st.selectbox('Thalassemia',
                    options=[3, 6, 7],
                    format_func=lambda x: {
                        3: 'Normal',
                        6: 'Fixed Defect',
                        7: 'Reversible Defect'
                    }[x])


input_dict = {
    'age': age,
    'sex': sex_val,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs_val,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang_val,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

input_df = pd.DataFrame([input_dict])

# Predict button
if st.button("Predict"):
    prediction = model_pipeline.predict(input_df)[0]

    if prediction == 1:
        st.error("High Risk: You may have heart disease. Consult a doctor.")
    else:
        st.success("Low Risk: No heart disease detected.")


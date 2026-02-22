import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")

st.title("❤️ Heart Disease Prediction System")

st.header("Enter Patient Details")

age = st.number_input("Age", 0, 100)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 250)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
restecg = st.selectbox("Resting ECG", [0,1,2])
thalach = st.number_input("Maximum Heart Rate", 60, 250)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("CA", [0,1,2,3,4])
thal = st.selectbox("Thal", [0,1,2,3])

if st.button("Predict"):
    data = [[age, sex, cp, trestbps, chol, fbs, restecg,
             thalach, exang, oldpeak, slope, ca, thal]]

    df = pd.DataFrame(data, columns=[
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"
    ])

    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({probability:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({probability:.2f}%)")
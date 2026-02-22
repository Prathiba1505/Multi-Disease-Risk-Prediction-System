import streamlit as st
import joblib
import pandas as pd

model = joblib.load("heart_model.pkl")

st.title("❤️ Heart Disease Prediction System")

st.header("Enter Patient Details")

age = st.number_input("Age", 0, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 250)
chol = st.number_input("Cholesterol", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar (0/1)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 250)
exang = st.selectbox("Exercise Induced Angina (0/1)", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("CA (0-4)", [0,1,2,3,4])
thal = st.selectbox("Thal (0-3)", [0,1,2,3])

if st.button("Predict"):
    
    data = pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,
                          thalach,exang,oldpeak,slope,ca,thal]],
                        columns=["age","sex","cp","trestbps","chol",
                                 "fbs","restecg","thalach","exang",
                                 "oldpeak","slope","ca","thal"])
    
    prediction = model.predict(data)
    probability = model.predict_proba(data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"Probability of Disease: {probability[0][1]*100:.2f}%")
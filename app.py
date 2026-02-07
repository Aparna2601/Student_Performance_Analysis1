import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Student Performance Analysis", page_icon="🎓", layout="centered")

# Load model
@st.cache_resource
def load_model():
    try:
        with open("models/model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

model = load_model()

# UI Header
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🎓Edu-Predict- Student Performance Analysis</h1>
    <p style='text-align: center;'>Predict whether a student will <b>PASS</b> or <b>FAIL</b> using Machine Learning</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Input Section
st.subheader("📋 Student Academic Details")

col1, col2 = st.columns(2)

with col1:
    math = st.number_input("Maths Marks", 0, 100, 60)
    science = st.number_input("Science Marks", 0, 100, 60)
    english = st.number_input("English Marks", 0, 100, 60)

with col2:
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study_hours = st.slider("Daily Study Hours", 0, 12, 4)
    assignments = st.slider("Assignments Completed (%)", 0, 100, 80)

# Performance indicators
average_marks = (math + science + english) / 3

st.subheader("📊 Performance Summary")
st.write(f"**Average Marks:** {average_marks:.2f}")
st.write(f"**Attendance:** {attendance}%")
st.write(f"**Study Hours:** {study_hours} hrs/day")

# Prediction
st.subheader("🔮 Future Performance Prediction")

if st.button("Predict Result"):
    if model is None:
        st.error("Model not found. Please train the model first.")
    else:
        input_df = pd.DataFrame([
            [math, science, english, attendance, study_hours, assignments]
        ], columns=[
            "math", "science", "english", "attendance", "study_hours", "assignments"
        ])

        prediction = model.predict(input_df)[0]

        if prediction >= 0.5:
            st.success("🎉 Prediction: **PASS**")
        else:
            st.error("⚠️ Prediction: **FAIL**")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size: 14px;'>
    Built using <b>Streamlit</b>, <b>DVC</b> & <b>MLflow</b> | MLOps Project – CSED2
    </p>
    """,
    unsafe_allow_html=True
)
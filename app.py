import streamlit as st
import pandas as pd
import joblib

# Load model and features only once
@st.cache_resource
def load_model_and_features():
    model = joblib.load('heart_disease_model_.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, feature_cols

model, feature_cols = load_model_and_features()

st.set_page_config(page_title="Heart Disease Risk Predictor")
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Fill in the patient details below to get a prediction.")

def user_input_features():
    data = {}

    # === NUMERIC INPUTS ===
    st.subheader("📏 Health Metrics")

    # ✅ BMI with empty placeholder and validation
    bmi_str = st.text_input("BMI", placeholder="Enter BMI (e.g. 23.5)")
    try:
        data['BMI'] = float(bmi_str) if bmi_str else None
    except ValueError:
        data['BMI'] = None
        st.warning("⚠️ Please enter a valid number for BMI.")

    # 👇 All other fields unchanged
    data['PhysicalHealth'] = st.number_input("Physical Health (days unwell past 30)", min_value=0, max_value=30)
    data['MentalHealth'] = st.number_input("Mental Health (days unwell past 30)", min_value=0, max_value=30)
    data['SleepTime'] = st.number_input("Average Sleep Time (hours)", min_value=0, max_value=24)

    # === BINARY QUESTIONS ===
    st.subheader("🏃 Lifestyle & History")
    binary_questions = {
        'Smoking': "Smoking",
        'AlcoholDrinking': "Alcohol Drinking",
        'Stroke': "History of Stroke",
        'DiffWalking': "Difficulty Walking",
        'Diabetic': "Diabetic",
        'PhysicalActivity': "Physical Activity",
        'Asthma': "Asthma",
        'KidneyDisease': "Kidney Disease",
        'SkinCancer': "Skin Cancer",
    }

    for key, label in binary_questions.items():
        response = st.selectbox(f"{label}", options=["Select", "Yes", "No"])
        if response == "Select":
            data[key] = None
        else:
            data[key] = 1 if response == "Yes" else 0

    # === GENHEALTH ===
    st.subheader("🩺 General Health")
    gen_health_map = {
        'Poor': 1, 'Fair': 2, 'Good': 3, 'Very Good': 4, 'Excellent': 5
    }
    gen_health_choice = st.selectbox("General Health", options=["Select"] + list(gen_health_map.keys()))
    data['GenHealth'] = gen_health_map.get(gen_health_choice, None)

    # === ONE-HOT ENCODED DEMOGRAPHICS ===
    st.subheader("🧍 Demographics")
    sex = st.selectbox("Sex", options=["Select", "Male", "Female"])
    race = st.selectbox("Race", options=[
        "Select", "American Indian/Alaskan Native", "Asian", "Black",
        "Hispanic", "Other", "White"
    ])
    age_category = st.selectbox("Age Category", options=[
        "Select", "18-24", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60-64", "65-69",
        "70-74", "75-79", "80 or older"
    ])

    for col in feature_cols:
        if col.startswith('Sex_') or col.startswith('Race_') or col.startswith('AgeCategory_'):
            data[col] = 0

    if sex != "Select":
        key = f"Sex_{sex.lower()}"
        if key in data:
            data[key] = 1
    else:
        return None, "Please select Sex."

    if race != "Select":
        key = f"Race_{race.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}"
        if key in data:
            data[key] = 1
    else:
        return None, "Please select Race."

    if age_category != "Select":
        key = f"AgeCategory_{age_category.replace(' ', '_').replace('/', '_').replace('-', '_')}"
        if key in data:
            data[key] = 1
    else:
        return None, "Please select Age Category."

    # Final check for missing values
    if None in data.values():
        return None, "Please fill out all fields before predicting."

    df = pd.DataFrame([data], columns=feature_cols)
    return df, None

# === GET USER INPUT ===
input_df, error_msg = user_input_features()

# === PREDICTION ===
if st.button("🔍 Predict Risk"):
    if error_msg:
        st.warning(f"⚠️ {error_msg}")
    elif input_df is not None:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error("⚠️ Patient is AT RISK of heart disease.")
        else:
            st.success("✅ Patient is NOT at risk of heart disease.")

        st.info(f"Confidence score: {proba:.2%}")

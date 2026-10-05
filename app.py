"""IT2011 Group 21 - Stroke risk screening app (course prototype, NOT for clinical use)."""
import math

import joblib
import pandas as pd
import streamlit as st

import iqr_capper  # noqa: F401  (the saved pipeline needs this class to load)

st.set_page_config(page_title="Stroke Risk Screening - IT2011 Group 21", page_icon="🧠", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.joblib")


bundle = load_bundle()
model, THRESHOLD, COLS = bundle["model"], bundle["threshold"], bundle["raw_columns"]

st.title("🧠 Stroke Risk Screening")
st.caption("IT2011 Group 21 · Calibrated XGBoost model trained on the Kaggle Stroke Prediction Dataset")
st.warning("Course prototype only - not a medical device. The result is a risk estimate, not a diagnosis.")

OUTPUT_MODES = {
    "Risk percentage": "percent",
    "Risk / No risk label": "label",
    "Both": "both",
}
mode_name = st.radio("Show the result as", list(OUTPUT_MODES), horizontal=True,
                     help="'Risk percentage' shows the estimated chance of stroke. "
                          "'Risk / No risk label' only shows whether the patient is above the screening threshold.")
mode = OUTPUT_MODES[mode_name]

with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age (years)", 1, 82, 67)
        gender = st.selectbox("Gender", ["Male", "Female"])
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart disease", ["No", "Yes"], horizontal=True)
        ever_married = st.radio("Ever married", ["Yes", "No"], horizontal=True)
    with c2:
        avg_glucose_level = st.number_input("Average glucose level (mg/dL)", 55.0, 272.0, 105.0, step=1.0)
        bmi_known = st.checkbox("BMI recorded", value=True)
        bmi = st.number_input("BMI", 10.0, 98.0, 28.0, step=0.1, disabled=not bmi_known)
        work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Residence type", ["Urban", "Rural"])
        smoking_status = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    patient = {
        "gender": gender, "age": float(age),
        "hypertension": int(hypertension == "Yes"), "heart_disease": int(heart_disease == "Yes"),
        "ever_married": ever_married, "work_type": work_type, "Residence_type": residence,
        "avg_glucose_level": float(avg_glucose_level),
        "bmi": float(bmi) if bmi_known else math.nan,
        "smoking_status": smoking_status,
    }
    risk = float(model.predict_proba(pd.DataFrame([patient])[COLS])[:, 1][0])
    at_risk = risk >= THRESHOLD
    label = "RISK" if at_risk else "NO RISK"

    st.subheader("Result")
    if mode in ("label", "both"):
        if at_risk:
            st.error(f"### ⚠️ {label}\nRisk is at or above the screening threshold - refer for follow-up.")
        else:
            st.success(f"### ✅ {label}\nRisk is below the screening threshold.")
    if mode in ("percent", "both"):
        m1, m2 = st.columns(2)
        m1.metric("Estimated stroke risk", f"{risk * 100:.1f} %")
        m2.metric("Screening threshold", f"{THRESHOLD * 100:.1f} %")
        st.progress(min(risk / 0.4, 1.0), text="Risk on a 0-40 % scale")

    with st.expander("How to read this"):
        st.markdown(
            f"""
- **Risk percentage** is the model's calibrated estimate of the chance that this patient has had a stroke.
  The average in the training data is **4.9 %**.
- **Risk / No risk** compares that percentage with the screening threshold (**{THRESHOLD * 100:.1f} %**).
  The threshold is deliberately low: the model is tuned to catch most strokes (about 70 % in testing),
  so many patients labelled "Risk" will not have had a stroke. Every "Risk" result needs a clinician's follow-up.
- Age has by far the largest effect. Smoking status was removed by feature selection, so it does not change the result.
"""
        )

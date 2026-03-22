#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Hero header */
.hero {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0;
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}
.section-label {
    color: #a78bfa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Input labels */
label {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Number inputs */
input[type=number] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 1rem !important;
}
input[type=number]:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
}

/* Primary button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
}

/* Result cards */
.result-diabetic {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-diabetic h2 { color: #f87171; font-size: 1.8rem; margin: 0; }

.result-healthy {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-healthy h2 { color: #34d399; font-size: 1.8rem; margin: 0; }

.confidence-badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 0.25rem 1rem;
    font-size: 0.95rem;
    color: #e2e8f0;
    margin-top: 0.5rem;
    font-weight: 500;
}
.result-note {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.75rem;
}

/* Summary table */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    border: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Prediction</h1>
    <p>AI-powered risk assessment based on clinical parameters</p>
</div>
""", unsafe_allow_html=True)

# ── Load model & scaler (cached) ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card"><div class="section-label">📋 Patient Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    pregnancies    = st.number_input('🤰 Pregnancies',               min_value=0,   max_value=20,  value=1,    step=1)
    glucose        = st.number_input('🩸 Glucose (mg/dL)',           min_value=0,   max_value=200, value=120,  step=1)
    blood_pressure = st.number_input('💓 Blood Pressure (mmHg)',     min_value=0,   max_value=150, value=70,   step=1)
    skin_thickness = st.number_input('📏 Skin Thickness (mm)',       min_value=0,   max_value=100, value=20,   step=1)

with col2:
    insulin        = st.number_input('💉 Insulin (mu U/ml)',         min_value=0,   max_value=900, value=80,   step=1)
    bmi            = st.number_input('⚖️ BMI',                       min_value=0.0, max_value=70.0,value=25.0, step=0.1, format="%.1f")
    dpf            = st.number_input('🧬 Diabetes Pedigree Function',min_value=0.0, max_value=2.5, value=0.5,  step=0.01,format="%.2f")
    age            = st.number_input('🎂 Age',                       min_value=1,   max_value=120, value=33,   step=1)

st.markdown('</div>', unsafe_allow_html=True)

# ── Validation ────────────────────────────────────────────────────────────────
def validate():
    issues = []
    if glucose == 0:      issues.append("Glucose cannot be 0.")
    if bmi == 0.0:        issues.append("BMI cannot be 0.")
    if blood_pressure == 0: issues.append("Blood Pressure cannot be 0.")
    return issues

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Diabetes Risk", type="primary"):
    issues = validate()
    if issues:
        for msg in issues:
            st.warning(f"⚠️ {msg}")
    else:
        input_data   = np.array([[pregnancies, glucose, blood_pressure,
                                   skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]
        confidence   = round(proba[prediction] * 100, 1)

        # Result card
        if prediction == 1:
            st.markdown(f"""
            <div class="result-diabetic">
                <h2>🔴 Diabetic</h2>
                <div class="confidence-badge">Confidence: {confidence}%</div>
                <p class="result-note">⚠️ High risk detected. Please consult a medical professional immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-healthy">
                <h2>🟢 Not Diabetic</h2>
                <div class="confidence-badge">Confidence: {confidence}%</div>
                <p class="result-note">✅ Low risk detected. Keep up the healthy lifestyle!</p>
            </div>
            """, unsafe_allow_html=True)

        # Metrics row
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Glucose",   f"{glucose} mg/dL")
        m2.metric("BMI",       f"{bmi}")
        m3.metric("Age",       f"{age} yrs")
        m4.metric("Insulin",   f"{insulin} µU/ml")

        # Summary table
        st.markdown("---")
        st.markdown('<div class="section-label">📊 Full Input Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Parameter": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                          "Insulin", "BMI", "Diabetes Pedigree Function", "Age"],
            "Value":     [pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, round(dpf, 2), age],
            "Unit":      ["—", "mg/dL", "mmHg", "mm", "µU/ml", "kg/m²", "—", "years"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.8rem;">'
    '⚕️ For educational purposes only. Not a substitute for professional medical advice.</p>',
    unsafe_allow_html=True
)

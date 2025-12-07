import streamlit as st
import json
import numpy as np
import onnxruntime as ort

# -----------------------------------
# Load ONNX Model & Feature Order
# -----------------------------------
st.set_page_config(
    page_title="Asthma Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_model():
    return ort.InferenceSession("asthma_rf.onnx")

@st.cache_resource
def load_features():
    with open("feature_order.json", "r") as f:
        return json.load(f)

model = load_model()
feature_order = load_features()

# -----------------------------------
# Styled UI
# -----------------------------------
st.markdown("""
<style>
.big-title {font-size:35px;font-weight:700;color:#256D85;}
.box { padding:12px; border-radius:10px; }
.green {background:#c8f7c5; border-left:6px solid #2ecc71;}
.yellow {background:#FFFACD; border-left:6px solid #F1C40F;}
.red {background:#ffb3b3; border-left:6px solid #e74c3c;}
.sidebar-anim {animation:pulse 2s infinite;}
@keyframes pulse{
    0%{background:#e8f6ff;}
    50%{background:#c1eaff;}
    100%{background:#e8f6ff;}
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='sidebar-anim'><h3>📌 How to Use</h3></div>", unsafe_allow_html=True)
    st.write("""
    👉 Enter your details  
    👉 Click Predict  
    👉 View your personalized risk level  
    ⚠️ Not for clinical diagnosis  
    """)

st.markdown("<p class='big-title'>🩺 Asthma Risk Assessment</p>", unsafe_allow_html=True)
st.write("Created by **Noel Graceson — AI Meets Healthcare**")

# -----------------------------------
# Mapping Functions (same as training)
# -----------------------------------

age_map = {"Under 15":0,"15-30":1,"30-45":2,"45-60":3,"60+":4}
binary_map = {"Yes":1, "No":0}
smoking_map = {"No":0, "Some days":1, "Every day":2, "Invalid":-1}
cigs_map = {"<1":0, ">5":1, "Invalid":-1}

def encode_duration(x):
    if x == "Invalid": return -1
    try:
        return int(float(str(x).split()[0]))
    except:
        return -1

# -----------------------------------
# UI Input Layout
# -----------------------------------
col1, col2 = st.columns(2)

with col1:
    age_group = st.selectbox("Age Group", list(age_map.keys()))
    pregnancy = st.selectbox("Pregnancy", ["Yes","No"])
    blood_pressure = st.selectbox("High Blood Pressure", ["Yes","No"])
    cholesterol = st.selectbox("High Cholesterol", ["Yes","No"])
    diabetes = st.selectbox("Diabetes", ["Yes","No"])
    home_pes = st.selectbox("Home Pesticides Exposure", ["Yes","No"])
    weed_pes = st.selectbox("Weed Pesticides Exposure", ["Yes","No"])
    had_asthma = st.selectbox("Ever Diagnosed with Asthma?", ["Yes","No"])

with col2:
    weight = st.number_input("Weight (kg)", 1.0, 200.0, 70.0)
    height = st.number_input("Height (cm)", 50.0, 220.0, 170.0)
    exercise = st.slider("Exercise Days per Month", 0, 30, 8)
    smoking = st.selectbox("Smoking Frequency", list(smoking_map.keys()))
    cigs = st.selectbox("Cigarettes per Day", list(cigs_map.keys()))
    duration_insulin = st.text_input("Insulin Duration (e.g., '6 months' or 'Invalid')", "Invalid")
    still_asthma = st.selectbox("Currently Have Asthma?", ["Yes","No"])
    er_visit = st.selectbox("ER Visit for Breathing Problems (Past year?)", ["Yes","No"])

# BMI calculation
bmi = weight / ((height/100)**2)

# -----------------------------------
# Build Feature Vector
# -----------------------------------
if st.button("🚀 Predict Asthma Risk"):
    
    data_dict = {
        "Age_Group": age_map[age_group],
        "Pregnancy_status": binary_map[pregnancy],
        "Blood_pressure": binary_map[blood_pressure],
        "Cholesterol": binary_map[cholesterol],
        "Diabetes": binary_map[diabetes],
        "Home_pesticides": binary_map[home_pes],
        "Weed_pesticides": binary_map[weed_pes],
        "Had_asthma": binary_map[had_asthma],
        "Still_asthma": binary_map[still_asthma],
        "ER_visit_past_year": binary_map[er_visit],
        "Smoking_frequency": smoking_map[smoking],
        "Cigarettes_per_day": cigs_map[cigs],
        "Duration_insulin": encode_duration(duration_insulin),
        "Weight_kg": weight,
        "Height_cm": height,
        "BMI": bmi,
        "Exercise_per_month": exercise,
    }

    # Order features exactly as training
    input_row = np.array([[data_dict[feat] for feat in feature_order]], dtype=np.float32)

    # Run ONNX prediction
    pred = model.run(None, {"input": input_row})[0][0]
    prob = pred

    # ----------------------------------------------------
    # Categorize risk using thresholds
    # ----------------------------------------------------
    if prob < 0.20:
        risk_class = "green"
        label = "NO RISK"
    elif prob < 0.50:
        risk_class = "yellow"
        label = "LOW RISK"
    else:
        risk_class = "red"
        label = "HIGH RISK"

    st.markdown(f"<div class='box {risk_class}'><h3>{label} ({prob:.2f})</h3></div>", unsafe_allow_html=True)

    st.subheader("📌 What This Means")
    st.write("""
    Your risk is influenced by:
    • Age & BMI  
    • Exposure & Smoking  
    • Past asthma history  
    • Insulin & comorbidities  
    
    This tool is for awareness only — not diagnosis.
    """)

    st.caption("Powered by ONNX AI • Built by Noel Graceson")


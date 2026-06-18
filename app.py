import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment Tool",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling (Glassmorphism, clean borders, professional fonts)
st.markdown("""
    <style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1E3A8A;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #4B5563;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 25px;
    }
    .card-container {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 25px;
        border: 1px solid #E5E7EB;
    }
    .low-risk-card {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
    }
    .medium-risk-card {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
    }
    .high-risk-card {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 5px 0;
    }
    .risk-label {
        font-weight: bold;
        text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load models and objects
@st.cache_resource
def load_ml_assets():
    model_path = "diabetes_model.pkl"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load models
model, scaler = load_ml_assets()

# Sidebar Layout
st.sidebar.image("https://img.icons8.com/illustrations/external-pack-flat-chamel-design/150/external-Medical-medical-care-pack-flat-chamel-design-2.png", width=120)
st.sidebar.markdown("### **Diabetes Risk Evaluator**")
st.sidebar.markdown(
    """
    This intelligent diagnostic support tool utilizes a **Random Forest Classifier** trained on the 
    **Pima Indians Diabetes Dataset** to predict the likelihood of diabetes mellitus.
    
    ### **Input Feature Descriptions**
    - **Pregnancies**: Number of times pregnant.
    - **Glucose**: 2-hour oral glucose tolerance test (mg/dL).
    - **Blood Pressure**: Diastolic blood pressure (mm Hg).
    - **Skin Thickness**: Triceps skin fold thickness (mm).
    - **Insulin**: 2-hour serum insulin (mu U/ml).
    - **BMI**: Body Mass Index (weight in kg / (height in m)²).
    - **Pedigree**: Diabetes pedigree function (genetic score based on family history).
    - **Age**: Age in years.
    """
)

st.sidebar.divider()
st.sidebar.caption("⚠️ **Disclaimer**: This tool is designed for educational purposes and medical internship training only. It does not constitute formal medical diagnostics.")

# Main Page Layout
st.markdown('<div class="main-header">Diabetes Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Input patient physiological parameters below to evaluate diabetes probability using our Machine Learning model.</div>', unsafe_allow_html=True)

if model is None or scaler is None:
    st.error("🚨 ML Model files (`diabetes_model.pkl` and `scaler.pkl`) not found! Please run the training script `train_model.py` first to generate them.")
else:
    # We organize the inputs in columns for a clean look
    st.subheader("📋 Patient Clinical Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Pregnancies", min_value=0, max_value=20, value=3, help="Number of times pregnant")
        glucose = st.number_input("Plasma Glucose Concentration (mg/dL)", min_value=0.0, max_value=300.0, value=117.0, step=1.0, help="2-hour oral glucose tolerance test (standard range: 70 - 140 mg/dL)")
        blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0.0, max_value=180.0, value=72.0, step=1.0, help="Normal diastolic range is 60 - 80 mm Hg")
        skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0.0, max_value=100.0, value=29.0, step=1.0, help="Used to estimate body fat percentage")
        
    with col2:
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0.0, max_value=1000.0, value=125.0, step=5.0, help="Standard range is heavily dependent on food intake")
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=80.0, value=32.4, step=0.1, help="Underweight < 18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese >= 30")
        pedigree = st.number_input("Diabetes Pedigree Score", min_value=0.0, max_value=3.0, value=0.38, step=0.01, help="Score representing genetic history of diabetes (higher is higher risk)")
        age = st.slider("Age (years)", min_value=1, max_value=120, value=29, help="Age of patient")
        
    st.markdown("---")
    
    # Predict button
    if st.button("Evaluate Diabetes Risk", type="primary", use_container_width=True):
        # 1. Arrange features into a 2D numpy array corresponding to columns:
        # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
        
        # 2. Scale features using the saved StandardScaler
        scaled_features = scaler.transform(features)
        
        # 3. Model Inference
        # Get binary prediction
        prediction = model.predict(scaled_features)[0]
        # Get probability of class 1 (Diabetic)
        probability = model.predict_proba(scaled_features)[0][1]
        
        # Determine Risk Level based on probability
        # Low: < 30%, Medium: 30% - 70%, High: > 70%
        if probability < 0.30:
            risk_level = "Low"
            card_class = "low-risk-card"
            text_color = "#10B981"
            guidance = "Based on the input physiological parameters, the model indicates a low likelihood of diabetes. Continue maintaining a balanced diet, staying physically active, and attending routine health checkups."
        elif 0.30 <= probability <= 0.70:
            risk_level = "Medium"
            card_class = "medium-risk-card"
            text_color = "#F59E0B"
            guidance = "The model suggests a moderate risk of diabetes. It is recommended to monitor blood glucose levels, maintain a healthy lifestyle, and consult a healthcare professional for standard screenings."
        else:
            risk_level = "High"
            card_class = "high-risk-card"
            text_color = "#EF4444"
            guidance = "The model indicates a high probability of diabetes. It is strongly advised to schedule a consultation with a physician or endocrinologist to perform diagnostic tests (A1C, Fasting Plasma Glucose) and receive professional evaluation."
            
        # Display color-coded card
        st.subheader("🔍 Assessment Results")
        
        st.markdown(f"""
            <div class="card-container {card_class}">
                <h4 style="margin: 0; color: #1F2937;">Model Evaluation: <span class="risk-label" style="color: {text_color};">{risk_level} Risk</span></h4>
                <div class="metric-value" style="color: {text_color};">{probability:.1%}</div>
                <p style="margin: 0 0 10px 0; color: #4B5563; font-weight: 500;">Probability of Diabetes Mellitus</p>
                <hr style="border: 0; border-top: 1px solid #D1D5DB; margin: 10px 0;">
                <p style="margin: 0; font-size: 0.95rem; color: #374151; line-height: 1.5;"><strong>Clinical Guidance:</strong> {guidance}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Let's show feature contributions or brief info
        with st.expander("ℹ️ How did the model make this decision?"):
            st.write(
                """
                Our **Random Forest** model evaluates all inputs simultaneously by passing them through a forest of 100 decision trees. 
                - **Glucose** levels are the strongest predictor (highest feature importance).
                - **BMI** and **Diabetes Pedigree Function** (genetic history) represent the secondary layer of decision boundaries.
                - When multiple key markers (such as High Glucose + High BMI + older Age) align, the individual tree votes converge towards a positive classification, resulting in a higher probability score.
                """
            )
            st.markdown(
                f"""
                **Technical Metrics:**
                - **Raw Inputs:** `Pregnancies={pregnancies}`, `Glucose={glucose} mg/dL`, `BloodPressure={blood_pressure} mm Hg`, `SkinThickness={skin_thickness} mm`, `Insulin={insulin} uU/ml`, `BMI={bmi}`, `Pedigree={pedigree:.3f}`, `Age={age}`
                - **Outcome Class Prediction:** `{prediction}` (where 1 = Positive, 0 = Negative)
                - **Classification Probability:** `{probability:.4f}`
                """
            )

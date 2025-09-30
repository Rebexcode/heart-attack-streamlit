import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

def main():
    st.set_page_config(
        page_title="Heart Attack Risk Assessment - Hypertensive Patients",
        page_icon="🩺",
        layout="centered"
    )
    
    st.title("Heart Attack Risk Assessment")
    st.markdown("### Specialized Model for Hypertensive Patients")
    st.info("⚠️ **Important**: This model is specifically designed and optimized for patients with hypertension (high blood pressure ≥140 mmHg). Accuracy is highest for this population.")
    st.markdown("Enter patient information to assess cardiovascular risk in hypertensive individuals")
    
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("Failed to load the prediction model. Please ensure model files are present.")
        return
    
    with st.expander("Understanding Hypertension & Heart Attack Risk"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Hypertension Categories:**
            - 🟢 **Normal**: <120 mmHg
            - 🟡 **Elevated**: 120-129 mmHg  
            - 🟠 **Stage 1**: 130-139 mmHg
            - 🔴 **Stage 2**: 140-159 mmHg
            - 🔴 **Severe**: 160-179 mmHg
            - 🚨 **Crisis**: ≥180 mmHg
            """)
        with col2:
            st.markdown("""
            **Why Hypertension Increases Heart Attack Risk:**
            - Damages artery walls over time
            - Forces heart to work harder
            - Increases plaque formation
            - Reduces blood flow to heart muscle
            - Often occurs with other risk factors
            """)

    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=50,
                help="Patient's age in years"
            )
            
            sex = st.selectbox(
                "Sex",
                options=[("Male", 1), ("Female", 0)],
                format_func=lambda x: x[0],
                help="Patient's biological sex"
            )[1]
            
            chest_pain = st.selectbox(
                "Chest Pain Type",
                options=[
                    ("Typical Angina", 1),
                    ("Atypical Angina", 2),
                    ("Non-anginal Pain", 3),
                    ("Asymptomatic", 4)
                ],
                format_func=lambda x: x[0],
                help="Type of chest pain experienced"
            )[1]
            
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=100,
                max_value=250,
                value=145,
                help="Resting systolic blood pressure in mm Hg. Model is optimized for hypertensive patients (≥130 mmHg)"
            )
            
            if resting_bp < 120:
                bp_category = "🟢 Normal"
                bp_warning = True
            elif resting_bp < 130:
                bp_category = "🟡 Elevated" 
                bp_warning = True
            elif resting_bp < 140:
                bp_category = "🟠 Stage 1 Hypertension"
                bp_warning = False
            elif resting_bp < 160:
                bp_category = "🔴 Stage 2 Hypertension"
                bp_warning = False
            elif resting_bp < 180:
                bp_category = "🔴 Severe Hypertension"
                bp_warning = False
            else:
                bp_category = "🚨 Hypertensive Crisis"
                bp_warning = False
            
            st.caption(f"Category: {bp_category}")
            
            if bp_warning and resting_bp < 130:
                st.warning("⚠️ This model is most accurate for hypertensive patients (BP ≥130). Results may be less reliable for normal/elevated BP.")
            
            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=100,
                max_value=600,
                value=200,
                help="Serum cholesterol level in mg/dl"
            )
            
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=[("False", 0), ("True", 1)],
                format_func=lambda x: x[0],
                help="Whether fasting blood sugar is greater than 120 mg/dl"
            )[1]
        
        with col2:
            resting_ecg = st.selectbox(
                "Resting Electrocardiogram Results",
                options=[
                    ("Normal", 0),
                    ("ST-T Wave Abnormality", 1),
                    ("Left Ventricular Hypertrophy", 2)
                ],
                format_func=lambda x: x[0],
                help="Results of resting electrocardiogram"
            )[1]
            
            max_hr = st.number_input(
                "Maximum Heart Rate Achieved",
                min_value=60,
                max_value=220,
                value=150,
                help="Maximum heart rate achieved during exercise"
            )
            
            exercise_angina = st.selectbox(
                "Exercise Induced Angina",
                options=[("No", 0), ("Yes", 1)],
                format_func=lambda x: x[0],
                help="Whether exercise induces angina"
            )[1]
            
            oldpeak = st.number_input(
                "Oldpeak (ST Depression)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="ST depression induced by exercise relative to rest"
            )
            
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=[
                    ("Upsloping", 1),
                    ("Flat", 2),
                    ("Downsloping", 3)
                ],
                format_func=lambda x: x[0],
                help="Slope of the peak exercise ST segment"
            )[1]
        
        submitted = st.form_submit_button("Predict Heart Attack Risk", use_container_width=True)
        
        if submitted:
            chest_pain_1 = 1 if chest_pain == 1 else 0
            chest_pain_2 = 1 if chest_pain == 2 else 0
            chest_pain_3 = 1 if chest_pain == 3 else 0
            chest_pain_4 = 1 if chest_pain == 4 else 0
            
            ecg_0 = 1 if resting_ecg == 0 else 0
            ecg_1 = 1 if resting_ecg == 1 else 0
            ecg_2 = 1 if resting_ecg == 2 else 0
            
            st_slope_0 = 1 if slope == 0 else 0
            st_slope_1 = 1 if slope == 1 else 0
            st_slope_2 = 1 if slope == 2 else 0
            st_slope_3 = 1 if slope == 3 else 0
            
            age_maxhr = age * max_hr / 100
            bp_chol = resting_bp * cholesterol / 1000
            maxhr_by_age = max_hr / age if age > 0 else 0
            angina_stdepression = exercise_angina * oldpeak
            
            feature_names = [
                'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHeartRate',
                'ExerciseAngina', 'STDepression', 'ChestPain_1', 'ChestPain_2', 
                'ChestPain_3', 'ChestPain_4', 'ECG_0', 'ECG_1', 'ECG_2', 
                'STSlope_0', 'STSlope_1', 'STSlope_2', 'STSlope_3', 
                'Age_MaxHR', 'BP_Chol', 'MaxHR_by_Age', 'Angina_STDepression'
            ]

            input_data = np.array([[
                age,
                sex,
                resting_bp,
                cholesterol,
                fasting_bs,
                max_hr,
                exercise_angina,
                oldpeak,
                chest_pain_1,
                chest_pain_2,
                chest_pain_3,
                chest_pain_4,
                ecg_0,
                ecg_1,
                ecg_2,
                st_slope_0,
                st_slope_1,
                st_slope_2,
                st_slope_3,
                age_maxhr,
                bp_chol,
                maxhr_by_age,
                angina_stdepression
            ]])
            
            try:
                input_df = pd.DataFrame(input_data, columns=feature_names)
                numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHeartRate', 
                                    'STDepression', 'Age_MaxHR', 'BP_Chol', 'MaxHR_by_Age', 
                                    'Angina_STDepression']
                input_df_scaled = input_df.copy()
                input_df_scaled[numerical_features] = scaler.transform(input_df[numerical_features])
                prediction = model.predict(input_df_scaled)[0]
                prediction_proba = model.predict_proba(input_df_scaled)[0]  
                confidence = max(prediction_proba) * 100
                
                risk_probability = prediction_proba[1] * 100 if len(prediction_proba) > 1 else prediction_proba[0] * 100
                
                if confidence >= 70:
                    certainty_level = "High"
                    certainty_color = "green"
                elif confidence >= 60:
                    certainty_level = "Moderate"
                    certainty_color = "orange"
                else:
                    certainty_level = "Moderate" 
                    certainty_color = "orange"
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                if prediction == 1:
                    st.error(f"⚠️ **HIGH RISK**: {risk_probability:.1f}% likelihood of heart attack")
                    risk_level = "High"
                    risk_color = "red"
                else:
                    st.success(f"✅ **LOW RISK**: {risk_probability:.1f}% likelihood of heart attack")
                    risk_level = "Low"
                    risk_color = "green"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Risk Level",
                        value=risk_level,
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Risk Probability",
                        value=f"{risk_probability:.1f}%",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence:.1f}%",
                        delta=None
                    )
                
                st.markdown("### Blood Pressure Assessment")
                
                if resting_bp >= 180:
                    urgency_level = "🚨 EMERGENCY"
                    urgency_color = "error"
                    urgency_msg = "Hypertensive crisis - seek immediate medical attention"
                elif resting_bp >= 160:
                    urgency_level = "🔴 URGENT"  
                    urgency_color = "error"
                    urgency_msg = "Severe hypertension - schedule medical evaluation within days"
                elif resting_bp >= 140:
                    urgency_level = "🟠 MODERATE"
                    urgency_color = "warning" 
                    urgency_msg = "Stage 2 hypertension - medical management needed"
                elif resting_bp >= 130:
                    urgency_level = "🟡 MILD"
                    urgency_color = "warning"
                    urgency_msg = "Stage 1 hypertension - lifestyle and/or medication management"
                else:
                    urgency_level = "🟢 MONITORING"
                    urgency_color = "info"
                    urgency_msg = "Pre-hypertensive - focus on prevention strategies"
                
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.metric(
                        label="BP Category", 
                        value=bp_category.replace("🟢 ", "").replace("🟡 ", "").replace("🟠 ", "").replace("🔴 ", "").replace("🚨 ", ""),
                        delta=f"{resting_bp} mmHg"
                    )
                with col2:
                    if urgency_color == "error":
                        st.error(f"**{urgency_level}**: {urgency_msg}")
                    elif urgency_color == "warning":
                        st.warning(f"**{urgency_level}**: {urgency_msg}")  
                    else:
                        st.info(f"**{urgency_level}**: {urgency_msg}")
                
                combined_risk = "Standard"
                if resting_bp >= 160 and risk_probability >= 50:
                    combined_risk = "Critical - Both severe hypertension AND high heart attack risk"
                elif resting_bp >= 140 and risk_probability >= 70:
                    combined_risk = "High - Stage 2 hypertension with elevated heart attack risk"
                elif resting_bp >= 130 and risk_probability >= 30:
                    combined_risk = "Elevated - Hypertension with notable heart attack risk"
                
                if combined_risk != "Standard":
                    st.error(f"**Combined Risk Assessment**: {combined_risk}")
                
                st.markdown("### 📋 Risk Interpretation for Hypertensive Patients")
                
                if risk_probability < 30:
                    st.info(f"""
                    **Low Risk ({risk_probability:.1f}%)**: Despite having hypertension, your other risk factors suggest a lower likelihood of heart attack. However, **hypertension management remains critical**:
                    
                    **Recommendations:**
                    - Continue blood pressure monitoring and medication compliance
                    - Maintain regular cardiology follow-ups
                    - Focus on lifestyle modifications (diet, exercise, stress management)
                    - Monitor for any new symptoms
                    """)
                elif risk_probability < 70:
                    st.warning(f"""
                    **Moderate Risk ({risk_probability:.1f}%)**: As a hypertensive patient with moderate risk, **active management is essential**:
                    
                    **Immediate Actions:**
                    - Schedule comprehensive cardiovascular evaluation within 2-4 weeks
                    - Review current BP medications with your doctor
                    - Consider cardiac stress testing or imaging
                    - Implement aggressive lifestyle changes
                    
                    **Management Focus:**
                    - Optimize blood pressure control (target <130/80)
                    - Add cardioprotective medications if appropriate
                    - Monitor other risk factors closely
                    """)
                else:
                    st.error(f"""
                    **High Risk ({risk_probability:.1f}%)**: **URGENT - This indicates very high cardiovascular risk in a hypertensive patient.**
                    
                    **Immediate Actions Required:**
                    - **Seek medical evaluation within 24-48 hours**
                    - Do not delay - contact your cardiologist or primary care provider immediately
                    - Consider emergency department if experiencing any chest pain, shortness of breath, or other cardiac symptoms
                    
                    **Expected Workup:**
                    - ECG, cardiac enzymes, and imaging studies
                    - Intensive blood pressure management
                    - Possible cardiac catheterization
                    - Aggressive medical therapy initiation
                    """)
                
                st.markdown("### Hypertension Management Priorities")
                hypertension_stage = ""
                if resting_bp < 130:
                    hypertension_stage = "Pre-hypertensive"
                elif resting_bp < 140:
                    hypertension_stage = "Stage 1"
                elif resting_bp < 160:
                    hypertension_stage = "Stage 2"
                elif resting_bp < 180:
                    hypertension_stage = "Severe"
                else:
                    hypertension_stage = "Crisis"
                
                st.info(f"""
                **Your BP Category**: {bp_category} ({hypertension_stage})
                
                **Key Management Points:**
                - **Target BP**: <130/80 mmHg for most hypertensive patients
                - **Medication**: Take prescribed antihypertensives consistently
                - **Monitoring**: Home BP monitoring recommended
                - **Lifestyle**: Regular exercise, DASH diet, sodium restriction
                - **Risk Factors**: Smoking cessation, weight management
                """)
                
                st.markdown("---")
                st.markdown("### ⚠️ Important Disclaimers & Limitations")
                
                st.error("""
                **CRITICAL MEDICAL DISCLAIMER:**
                - This tool is for **educational purposes ONLY** and **cannot replace professional medical evaluation**
                - **Do not use for emergency medical decisions** - if experiencing chest pain, shortness of breath, or other cardiac symptoms, seek immediate medical care
                - Results are **most accurate for hypertensive patients** (the model's training population)
                """)
                
                st.warning("""
                **Model Limitations:**
                - **Optimized for hypertensive patients**: 83.6% of training data had high blood pressure
                - **Less reliable for normal BP**: Only 16.4% of training data had normal/elevated BP
                - **Population bias**: Training data heavily weighted toward cardiovascular risk patients
                - **Not validated** on all ethnic groups or age ranges equally
                """)
                
                st.info("""
                **Always Consult Healthcare Professionals For:**
                - 🩺 Blood pressure management and medication adjustments
                - 📋 Comprehensive cardiovascular risk assessment  
                - 🔍 Diagnostic testing (stress tests, imaging, etc.)
                - 💊 Treatment planning and medication decisions
                - 🚨 Any concerning symptoms or changes in condition
                """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check that all input values are valid and try again.")

if __name__ == "__main__":
    main()
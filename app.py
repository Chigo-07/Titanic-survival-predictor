import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# --- 1. Load Model with Error Handling ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model/titanic_survival_model.pkl')
        # We manually map sex to avoid dependency on loading a second file if possible, 
        # but loading the encoder is safer if mapping changes.
        # For simplicity/robustness here, we will handle mapping logic in code.
        return model
    except FileNotFoundError:
        st.error("❌ Critical Error: Model file not found. Please check 'model/titanic_survival_model.pkl'.")
        return None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading the model: {e}")
        return None

model = load_resources()

# --- UI Header ---
st.title("🚢 Titanic Survival Prediction System")
st.markdown("Enter passenger details to predict their survival probability.")

if model:
    # --- 2. User Inputs ---
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else (f"{x}nd Class" if x==2 else f"{x}rd Class"))
        sex = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 0, 100, 30)

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare Price (£)", min_value=0.0, value=15.0)

    # --- Preprocessing for Prediction ---
    # Must match the training phase exactly
    sex_encoded = 1 if sex == "Male" else 0
    
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Fare': [fare]
    })

    # --- 3. Prediction with Error Handling ---
    if st.button("Predict Survival", type="primary"):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            st.divider()
            
            if prediction == 1:
                confidence = probability[1] * 100
                st.success(f"**Prediction: SURVIVED** 🎉")
                st.info(f"The model is **{confidence:.1f}%** confident in this prediction.")
            else:
                confidence = probability[0] * 100
                st.error(f"**Prediction: DID NOT SURVIVE** 💀")
                st.info(f"The model is **{confidence:.1f}%** confident in this prediction.")

        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")
            st.write("Please check input values or contact the administrator.")

else:
    st.warning("The application cannot run because the model failed to load.")

# Footer
st.markdown("---")
st.caption("Titanic Project | Developed by [Your Name]")

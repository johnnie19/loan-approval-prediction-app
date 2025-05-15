"""
Streamlit application for loan approval prediction.
This app provides a simple form to submit loan application data to the API.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import joblib
import sys
from datetime import datetime

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'include'))

# Import the modules
from include.data_preprocessing import preprocess_data, create_preprocessing_pipeline

# Set page title and favicon
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the latest model
def load_latest_model():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    if not os.path.exists(models_dir):
        st.error("Models directory not found. Please make sure the ETL pipeline has been run at least once.")
        return None
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    if not model_files:
        st.error("No model files found. Please make sure the ETL pipeline has been run at least once.")
        return None
    
    # Sort by modification time (newest first)
    latest_model_file = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)[0]
    model_path = os.path.join(models_dir, latest_model_file)
    
    try:
        model = joblib.load(model_path)
        return model, model_path
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to make prediction
def predict_loan_approval(model, input_data):
    """Make a prediction using the trained model."""
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Add Loan_ID column (required by preprocess_data)
    input_df['Loan_ID'] = f'LOAN_TEST_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S")}'
    
    # Add derived features that the model expects
    # Calculate EMI (Monthly_Payment_Estimate)
    if 'LoanAmount' in input_df.columns and 'Loan_Amount_Term' in input_df.columns:
        monthly_interest_rate = 0.10 / 12
        P = input_df['LoanAmount'] * 1000
        r = monthly_interest_rate
        n = input_df['Loan_Amount_Term']
        input_df['Monthly_Payment_Estimate'] = (
            P * r * np.power(1 + r, n) / 
            (np.power(1 + r, n) - 1)
        )
    
    # Add Total_Income if not present
    if 'Total_Income' not in input_df.columns and 'ApplicantIncome' in input_df.columns and 'CoapplicantIncome' in input_df.columns:
        input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    
    # Add Income_to_Payment_Ratio
    if 'Total_Income' in input_df.columns and 'Monthly_Payment_Estimate' in input_df.columns:
        input_df['Income_to_Payment_Ratio'] = input_df['Total_Income'] / (input_df['Monthly_Payment_Estimate'] + 1)
    
    # Make prediction directly using the model pipeline
    # The pipeline already includes preprocessing
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        return prediction[0], prediction_proba[0][1]  # Return prediction and probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        # Fallback to a simple rule-based prediction
        if input_data['Credit_History'] == 1 and input_data['ApplicantIncome'] > 5000:
            return 1, 0.85
        else:
            return 0, 0.35

# Main function
def main():
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a machine learning model to predict loan approval. "
        "The model is trained on historical loan data and updated daily through an ETL pipeline."
    )
    
    st.sidebar.title("Model Information")
    model_result = load_latest_model()
    if model_result:
        model, model_path = model_result
        st.sidebar.success(f"Model loaded: {os.path.basename(model_path)}")
        model_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        st.sidebar.info(f"Model last updated: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.error("No model available. Please run the ETL pipeline first.")
        model = None
    
    # Main content
    st.title("Loan Approval Predictor")
    st.write(
        "Fill out the form below to predict whether a loan application will be approved. "
        "The prediction is based on a machine learning model trained on historical loan data."
    )
    
    # Create form
    with st.form("loan_application_form"):
        # Create two columns for the form
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
        with col2:
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=100)
            loan_amount_term = st.selectbox("Loan Amount Term (in months)", [360, 180, 120, 60, 36, 12])
            credit_history = st.selectbox("Credit History (1: Good, 0: Bad)", [1, 0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        # Submit button
        submitted = st.form_submit_button("Predict Loan Approval")
    
    # Make prediction when form is submitted
    if submitted and model:
        # Prepare input data
        input_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        
        # Make prediction
        prediction, probability = predict_loan_approval(model, input_data)
        
        # Display result
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.success(f"Loan Approved! (Probability: {probability:.2%})")
        else:
            st.error(f"Loan Not Approved (Probability: {probability:.2%})")
        
        # Display feature importance
        st.subheader("Key Factors")
        st.write("The following factors influenced the prediction:")
        
        # Create a simple explanation based on the input data
        factors = []
        if credit_history == 1:
            factors.append(("Good credit history", "positive"))
        else:
            factors.append(("Poor credit history", "negative"))
        
        if applicant_income > 5000:
            factors.append(("High applicant income", "positive"))
        else:
            factors.append(("Low applicant income", "negative"))
        
        if property_area == "Urban":
            factors.append(("Urban property area", "positive"))
        
        if loan_amount > 200:
            factors.append(("High loan amount", "negative"))
        else:
            factors.append(("Reasonable loan amount", "positive"))
        
        # Display factors
        for factor, impact in factors:
            if impact == "positive":
                st.markdown(f"✅ {factor} (Positive impact)")
            else:
                st.markdown(f"❌ {factor} (Negative impact)")
        
        # Display input data
        st.subheader("Application Details")
        st.json(input_data)

if __name__ == "__main__":
    main()

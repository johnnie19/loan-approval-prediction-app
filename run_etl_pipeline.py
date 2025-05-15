#!/usr/bin/env python3
"""
ETL Pipeline Runner

This script runs the ETL pipeline for loan approval prediction without using Airflow:
1. Simulates daily arrival of new data
2. Loads and preprocesses the data
3. Trains a machine learning model
4. Logs the model and metrics
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add the include directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(current_dir, 'include')
sys.path.append(include_dir)

# Import custom modules
from data_preprocessing import get_preprocessed_data
from train_model import train_and_evaluate

# Define paths
DATA_DIR = os.path.join(current_dir, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(current_dir, 'models')
LOGS_DIR = os.path.join(current_dir, 'logs')

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def simulate_new_data():
    """
    Simulate the arrival of new loan application data.
    In a real-world scenario, this would be replaced with actual data ingestion.
    """
    print("Simulating new data arrival...")
    
    # Create a timestamp for the data file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RAW_DATA_DIR, f"loan_data_{timestamp}.csv")
    
    # Generate synthetic data
    n_samples = 100
    
    # Define column names and data types
    columns = {
        'Loan_ID': [f'LOAN_{i:06d}' for i in range(1, n_samples + 1)],
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
        'Married': np.random.choice(['Yes', 'No'], size=n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], size=n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], size=n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], size=n_samples),
        'ApplicantIncome': np.random.randint(1000, 10000, size=n_samples),
        'CoapplicantIncome': np.random.randint(0, 5000, size=n_samples),
        'LoanAmount': np.random.randint(50, 500, size=n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 120, 60, 36, 12], size=n_samples),
        'Credit_History': np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], size=n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(columns)
    
    # Add target variable (loan status) based on some rules to make it realistic
    # Higher income, good credit history, and urban property increase approval chances
    approval_score = (
        (df['ApplicantIncome'] > 5000).astype(int) * 2 +
        (df['CoapplicantIncome'] > 2000).astype(int) +
        (df['Credit_History'] == 1).astype(int) * 3 +
        (df['Property_Area'] == 'Urban').astype(int) * 2 +
        (df['Education'] == 'Graduate').astype(int) +
        (df['LoanAmount'] < 200).astype(int)
    )
    
    # Convert score to binary outcome with some randomness
    threshold = 5
    random_factor = np.random.normal(0, 1, n_samples)
    df['Loan_Status'] = ((approval_score + random_factor) > threshold).map({True: 'Y', False: 'N'})
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated synthetic data with {n_samples} samples and saved to {output_file}")
    
    # Return the file path for the next task
    return output_file

def preprocess_data(raw_data_file):
    """
    Load and preprocess the raw data.
    """
    print("Preprocessing data...")
    
    # Create a timestamp for the processed data file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_data_file = os.path.join(PROCESSED_DATA_DIR, f"processed_loan_data_{timestamp}.csv")
    
    # Load and preprocess the data
    X, y, preprocessor = get_preprocessed_data(raw_data_file)
    
    # Combine features and target for saving
    processed_df = X.copy()
    processed_df['Loan_Status'] = y
    
    # Save the processed data
    processed_df.to_csv(processed_data_file, index=False)
    print(f"Preprocessed data saved to {processed_data_file}")
    
    # Return the processed data file path for the next task
    return processed_data_file

def train_model(processed_data_file):
    """
    Train a machine learning model on the preprocessed data.
    """
    print("Training model...")
    
    # Set model parameters
    model_type = 'random_forest'
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
    
    # Train and evaluate the model
    model, metrics, model_path = train_and_evaluate(
        data_path=processed_data_file,
        output_dir=MODELS_DIR,
        model_type=model_type,
        params=model_params
    )
    
    print(f"Model saved to {model_path}")
    
    # Return the model path and metrics for the next task
    return {
        'model_path': model_path,
        'metrics': metrics
    }

def log_results(model_info):
    """
    Log the results of the model training.
    """
    print("Logging results...")
    
    if not model_info:
        print("No model information available. Training may have failed.")
        return
    
    model_path = model_info.get('model_path')
    metrics = model_info.get('metrics', {})
    
    # Log the results
    print(f"Model saved to: {model_path}")
    print("Model Performance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Create a timestamp for the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model info to a file for the Streamlit app to use
    model_info_file = os.path.join(MODELS_DIR, "model_info.txt")
    with open(model_info_file, 'w') as f:
        f.write(f"Model path: {model_path}\n")
        f.write(f"Last updated: {timestamp}\n")
        f.write("Metrics:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")
    
    # Also save to logs directory with timestamp
    log_file = os.path.join(LOGS_DIR, f"model_training_log_{timestamp}.txt")
    with open(log_file, 'w') as f:
        f.write(f"Model path: {model_path}\n")
        f.write(f"Training completed: {timestamp}\n")
        f.write("Metrics:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")
    
    print(f"Model info saved to: {model_info_file}")
    print(f"Training log saved to: {log_file}")

def run_etl_pipeline():
    """
    Run the complete ETL pipeline.
    """
    print("Starting ETL pipeline...")
    
    # Step 1: Simulate new data
    raw_data_file = simulate_new_data()
    
    # Step 2: Preprocess data
    processed_data_file = preprocess_data(raw_data_file)
    
    # Step 3: Train model
    model_info = train_model(processed_data_file)
    
    # Step 4: Log results
    log_results(model_info)
    
    print("ETL pipeline completed successfully!")

if __name__ == "__main__":
    run_etl_pipeline()

"""
ETL Pipeline DAG for Loan Approval Prediction

This DAG implements a complete ETL pipeline for loan approval prediction:
1. Simulates daily arrival of new data
2. Loads and preprocesses the data
3. Trains a machine learning model
4. Logs the model and metrics
"""

from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import joblib
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

# Import custom modules
import sys
sys.path.append('/opt/airflow/include')
from data_preprocessing import get_preprocessed_data
from train_model import train_and_evaluate

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'loan_approval_etl_pipeline',
    default_args=default_args,
    description='ETL Pipeline for Loan Approval Prediction',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=days_ago(1),
    catchup=False,
    tags=['etl', 'ml', 'loan_approval'],
)

# Define paths
DATA_DIR = '/opt/airflow/data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'
MODELS_DIR = '/opt/airflow/models'

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Task 1: Simulate new data arrival
def simulate_new_data(**kwargs):
    """
    Simulate the arrival of new loan application data.
    In a real-world scenario, this would be replaced with actual data ingestion.
    """
    # Create a timestamp for the data file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{RAW_DATA_DIR}/loan_data_{timestamp}.csv"
    
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

# Task 2: Preprocess data
def preprocess_data(**kwargs):
    """
    Load and preprocess the raw data.
    """
    # Get the raw data file path from the previous task
    ti = kwargs['ti']
    raw_data_file = ti.xcom_pull(task_ids='simulate_new_data')
    
    # Create a timestamp for the processed data file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_data_file = f"{PROCESSED_DATA_DIR}/processed_loan_data_{timestamp}.csv"
    
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

# Task 3: Train model
def train_model(**kwargs):
    """
    Train a machine learning model on the preprocessed data.
    """
    # Get the processed data file path from the previous task
    ti = kwargs['ti']
    processed_data_file = ti.xcom_pull(task_ids='preprocess_data')
    
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
    
    # Return the model path and metrics for the next task
    return {
        'model_path': model_path,
        'metrics': metrics
    }

# Task 4: Log results
def log_results(**kwargs):
    """
    Log the results of the model training.
    """
    # Get the model information from the previous task
    ti = kwargs['ti']
    model_info = ti.xcom_pull(task_ids='train_model')
    
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
    model_info_file = f"{MODELS_DIR}/model_info.txt"
    with open(model_info_file, 'w') as f:
        f.write(f"Model path: {model_path}\n")
        f.write(f"Last updated: {timestamp}\n")
        f.write("Metrics:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")
    
    print(f"Model info saved to: {model_info_file}")
    
    # In a real-world scenario, you might want to:
    # - Send notifications about model performance
    # - Update a dashboard or report
    # - Trigger model deployment if metrics meet certain criteria

# Define the tasks
task_simulate_data = PythonOperator(
    task_id='simulate_new_data',
    python_callable=simulate_new_data,
    dag=dag,
)

task_preprocess_data = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

task_train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

task_log_results = PythonOperator(
    task_id='log_results',
    python_callable=log_results,
    dag=dag,
)

# Set task dependencies
task_simulate_data >> task_preprocess_data >> task_train_model >> task_log_results

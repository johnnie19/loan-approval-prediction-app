"""
Script to generate a sample model file for the Streamlit app to use.
This is useful for demonstration purposes without having to run the full ETL pipeline.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

# Create directories if they don't exist
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(models_dir, exist_ok=True)

# Generate a sample dataset
def generate_sample_data(n_samples=1000):
    """Generate a sample loan dataset."""
    np.random.seed(42)
    
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
        'Loan_Status': np.random.choice(['Y', 'N'], size=n_samples, p=[0.7, 0.3]),
    }
    
    # Create DataFrame
    df = pd.DataFrame(columns)
    return df

# Preprocess the data
def preprocess_data(df, target_column='Loan_Status'):
    """Preprocess the data for model training."""
    # Convert target to binary
    if target_column:
        y = df[target_column].map({'Y': 1, 'N': 0})
        X = df.drop(columns=[target_column, 'Loan_ID'])
    else:
        y = None
        X = df.drop(columns=['Loan_ID'])
    
    return X, y

# Create a preprocessing pipeline
def create_preprocessing_pipeline(X):
    """Create a preprocessing pipeline for the data."""
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing steps for numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

# Train a model
def train_model(X, y):
    """Train a random forest model."""
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X)
    
    # Create and train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)
    return model

# Save the model
def save_model(model, filename):
    """Save the model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main():
    """Main function to generate and save a sample model."""
    print("Generating sample model...")
    
    # Generate sample data
    df = generate_sample_data()
    
    # Preprocess the data
    X, y = preprocess_data(df)
    
    # Train the model
    model = train_model(X, y)
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = os.path.join(models_dir, f'loan_approval_model_{timestamp}.joblib')
    save_model(model, model_filename)
    
    print("Sample model generation complete!")

if __name__ == "__main__":
    main()

"""
Test script for the ETL pipeline components.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'include'))

# Import the modules to test
from include.data_preprocessing import load_data, preprocess_data, create_preprocessing_pipeline
from include.train_model import create_model_pipeline, evaluate_model

# Import the DAG functions
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dags'))
from dags.etl_pipeline import simulate_new_data


class TestETLPipeline(unittest.TestCase):
    """Test cases for the ETL pipeline components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a sample dataset for testing
        self.sample_data_path = os.path.join(self.test_dir, 'sample_loan_data.csv')
        self.create_sample_dataset()

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test files
        if os.path.exists(self.sample_data_path):
            os.remove(self.sample_data_path)
        
        # Remove the test directory if it's empty
        if os.path.exists(self.test_dir) and not os.listdir(self.test_dir):
            os.rmdir(self.test_dir)

    def create_sample_dataset(self):
        """Create a sample dataset for testing."""
        # Generate a small sample dataset
        n_samples = 20
        
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
            'Loan_Status': np.random.choice(['Y', 'N'], size=n_samples),
        }
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(columns)
        df.to_csv(self.sample_data_path, index=False)

    def test_data_preprocessing(self):
        """Test the data preprocessing functions."""
        # Test loading data
        df = load_data(self.sample_data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 20)  # Check that all rows were loaded
        
        # Test preprocessing
        X, y = preprocess_data(df)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        
        # Test preprocessing pipeline creation
        preprocessor = create_preprocessing_pipeline(X)
        self.assertIsNotNone(preprocessor)

    def test_model_creation(self):
        """Test the model creation functions."""
        # Test creating a model pipeline
        model = create_model_pipeline(model_type='random_forest')
        self.assertIsNotNone(model)
        
        # Test with parameters
        params = {'n_estimators': 50, 'max_depth': 5}
        model = create_model_pipeline(model_type='random_forest', params=params)
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 5)

    def test_simulate_new_data(self):
        """Test the simulate_new_data function."""
        # Mock the kwargs for the function
        kwargs = {}
        
        # Override the output directory
        import dags.etl_pipeline as etl
        original_raw_data_dir = etl.RAW_DATA_DIR
        etl.RAW_DATA_DIR = self.test_dir
        
        try:
            # Call the function
            output_file = simulate_new_data(**kwargs)
            
            # Check that the file was created
            self.assertTrue(os.path.exists(output_file))
            
            # Check that the file contains data
            df = pd.read_csv(output_file)
            self.assertGreater(len(df), 0)
            self.assertIn('Loan_Status', df.columns)
        finally:
            # Restore the original directory
            etl.RAW_DATA_DIR = original_raw_data_dir
            
            # Clean up the generated file
            if os.path.exists(output_file):
                os.remove(output_file)


if __name__ == '__main__':
    unittest.main()

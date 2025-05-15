import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load the dataset and perform initial cleaning"""
    df = pd.read_csv(file_path)
    
    # Remove 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    return df

def preprocess_data(df, target_column='Loan_Status'):
    """Preprocess data for modeling"""
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # ---- Handle Total_Income with currency formatting ----
    if 'Total_Income' in df_processed.columns and df_processed['Total_Income'].dtype == 'object':
        # Remove currency symbols, commas, and other non-numeric characters except decimal points
        df_processed['Total_Income'] = df_processed['Total_Income'].replace('[\$,]', '', regex=True)
        # Convert to numeric
        df_processed['Total_Income'] = pd.to_numeric(df_processed['Total_Income'], errors='coerce')
    
    # ---- Handle Dependents column ----
    if 'Dependents' in df_processed.columns:
        df_processed['Dependents'] = df_processed['Dependents'].replace('3+', '3')
        df_processed['Dependents'] = pd.to_numeric(df_processed['Dependents'], errors='coerce')
    
    # ---- Feature Engineering ----
    # Calculate Total_Income if missing
    if ('Total_Income' in df_processed.columns and 
        'ApplicantIncome' in df_processed.columns and 
        'CoapplicantIncome' in df_processed.columns):
        if df_processed['Total_Income'].isnull().all() or df_processed['Total_Income'].isnull().sum() > df_processed.shape[0] * 0.5:
            df_processed['Total_Income'] = df_processed['ApplicantIncome'] + df_processed['CoapplicantIncome']
    
    # Add EMI calculation
    if 'LoanAmount' in df_processed.columns and 'Loan_Amount_Term' in df_processed.columns:
        monthly_interest_rate = 0.10 / 12
        P = df_processed['LoanAmount'] * 1000
        r = monthly_interest_rate
        n = df_processed['Loan_Amount_Term']
        valid_mask = (n > 0) & n.notna()
        df_processed['Monthly_Payment_Estimate'] = np.nan
        df_processed.loc[valid_mask, 'Monthly_Payment_Estimate'] = (
            P[valid_mask] * r * np.power(1 + r, n[valid_mask]) / 
            (np.power(1 + r, n[valid_mask]) - 1)
        )
    
    # Add affordability ratio
    if 'Total_Income' in df_processed.columns and 'Monthly_Payment_Estimate' in df_processed.columns:
        df_processed['Income_to_Payment_Ratio'] = df_processed['Total_Income'] / (df_processed['Monthly_Payment_Estimate'] + 1)
    
    # ---- Prepare for ML pipeline ----
    # Separate features and target
    if target_column in df_processed.columns:
        X = df_processed.drop(target_column, axis=1)
        y = df_processed[target_column]
        if y.dtype == 'object':
            # Convert Y/N to 1/0 for binary classification
            y = y.map({'Y': 1, 'N': 0})
    else:
        X = df_processed
        y = None
    
    # Drop ID column if present
    if 'Loan_ID' in X.columns:
        X = X.drop('Loan_ID', axis=1)
    
    return X, y

def create_preprocessing_pipeline(X):
    """Create a preprocessing pipeline for numerical and categorical features"""
    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def get_preprocessed_data(file_path, target_column='Loan_Status'):
    """Load, preprocess, and prepare data for modeling in one function"""
    df = load_data(file_path)
    X, y = preprocess_data(df, target_column)
    preprocessor = create_preprocessing_pipeline(X)
    
    return X, y, preprocessor

if __name__ == "__main__":
    # Example usage
    file_path = "df1_loan.csv"
    X, y, preprocessor = get_preprocessed_data(file_path)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape if y is not None else 'No target'}")
    print(f"Number of numerical features: {len(X.select_dtypes(include=['int64', 'float64']).columns)}")
    print(f"Number of categorical features: {len(X.select_dtypes(include=['object']).columns)}")

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# Optional: Uncomment if xgboost or lightgbm are installed
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import preprocessing module - this will be handled by Airflow's import system
# from data_preprocessing import get_preprocessed_data

def create_model_pipeline(model_type='random_forest', params=None):
    """Create a pipeline with the specified model"""
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(random_state=42, probability=True),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'knn': KNeighborsClassifier(),
        'naive_bayes': GaussianNB(),
        'extra_trees': ExtraTreesClassifier(random_state=42),
        # Optional: Uncomment if xgboost or lightgbm are installed
        # 'xgboost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        # 'lightgbm': LGBMClassifier(random_state=42)
    }
    
    if model_type not in models:
        raise ValueError(f"Model type '{model_type}' not supported. Choose from: {list(models.keys())}")
    
    model = models[model_type]
    
    # Apply custom parameters if provided
    if params:
        model.set_params(**params)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with multiple metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # In Airflow environment, we'll skip the visualization parts
    # but keep the code commented for reference
    
    # # Print confusion matrix
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=['Not Approved', 'Approved'],
    #             yticklabels=['Not Approved', 'Approved'])
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # plt.tight_layout()
    # plt.show()
    
    # # ROC curve if probabilities are available
    # if y_pred_proba is not None:
    #     fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
    #     plt.plot([0, 1], [0, 1], 'k--', label='Random')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('ROC Curve')
    #     plt.legend()
    #     plt.grid(alpha=0.3)
    #     plt.show()
    
    return metrics

def get_feature_importance(model, features, top_n=10):
    """Extract feature importance from the model if available"""
    # Check if the model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Debug information
        print(f"Number of features in model: {len(importances)}")
        print(f"Number of feature names provided: {len(features)}")
        
        # Make sure lengths match
        if len(importances) != len(features):
            print("WARNING: Feature count mismatch. This often happens with categorical encoding.")
            print("Some feature names may be incorrect or missing.")
            
            # Create generic feature names if needed
            if len(importances) > len(features):
                print(f"Creating generic names for {len(importances) - len(features)} additional features")
                # Extend feature list with generic names
                features = list(features) + [f"feature_{i}" for i in range(len(features), len(importances))]
            else:
                # Truncate feature list to match importances
                print(f"Truncating feature list from {len(features)} to {len(importances)}")
                features = list(features)[:len(importances)]
        
        # Create DataFrame with matching lengths
        feature_importance = pd.DataFrame({'feature': features, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
        
        # In Airflow environment, we'll skip the visualization parts
        # but keep the code commented for reference
        
        # # Plot feature importance
        # plt.figure(figsize=(10, 6))
        # sns.barplot(x='importance', y='feature', data=feature_importance)
        # plt.title(f'Top {top_n} Feature Importance')
        # plt.tight_layout()
        # plt.show()
        
        return feature_importance
    
    # Check if model has coef_ attribute (linear models)
    elif hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        
        # Debug information
        print(f"Number of coefficients in model: {len(coefficients)}")
        print(f"Number of feature names provided: {len(features)}")
        
        # Make sure lengths match
        if len(coefficients) != len(features):
            print("WARNING: Feature count mismatch. This often happens with categorical encoding.")
            print("Some feature names may be incorrect or missing.")
            
            # Create generic feature names if needed
            if len(coefficients) > len(features):
                print(f"Creating generic names for {len(coefficients) - len(features)} additional features")
                # Extend feature list with generic names
                features = list(features) + [f"feature_{i}" for i in range(len(features), len(coefficients))]
            else:
                # Truncate feature list to match coefficients
                print(f"Truncating feature list from {len(features)} to {len(coefficients)}")
                features = list(features)[:len(coefficients)]
        
        # Create DataFrame with matching lengths
        feature_importance = pd.DataFrame({'feature': features, 'coefficient': np.abs(coefficients)})
        feature_importance = feature_importance.sort_values('coefficient', ascending=False).head(top_n)
        
        # In Airflow environment, we'll skip the visualization parts
        # but keep the code commented for reference
        
        # # Plot feature importance
        # plt.figure(figsize=(10, 6))
        # sns.barplot(x='coefficient', y='feature', data=feature_importance)
        # plt.title(f'Top {top_n} Feature Importance (Absolute Coefficients)')
        # plt.tight_layout()
        # plt.show()
        
        return feature_importance
    
    else:
        print("Feature importance not available for this model type")
        return None

def tune_hyperparameters(X, y, model_type, param_grid, cv=10):
    """Tune model hyperparameters using GridSearchCV"""
    # Create base model
    model = create_model_pipeline(model_type)
    
    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search to the data
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, output_dir, filename="loan_approval_model", add_timestamp=True):
    """
    Save the model to the specified directory
    
    Parameters:
    -----------
    model : object
        The trained model to save
    output_dir : str
        Directory where the model will be saved
    filename : str
        Base name for the model file
    add_timestamp : bool
        Whether to add a timestamp to the filename
    
    Returns:
    --------
    str
        The full path where the model was saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the full path with optional timestamp
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(output_dir, f"{filename}_{timestamp}.joblib")
    else:
        full_path = os.path.join(output_dir, f"{filename}.joblib")

    # Save the model with error handling and logging
    try:
        joblib.dump(model, full_path)
        print(f"Model saved to {full_path}")
    except Exception as e:
        print(f"Error saving model to {full_path}: {e}")
        raise
    return full_path

def train_and_evaluate(data_path, output_dir, model_type='random_forest', params=None, test_size=0.2):
    """Complete pipeline to train and evaluate a model"""
    # This function assumes get_preprocessed_data is imported from data_preprocessing
    from data_preprocessing import get_preprocessed_data
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, preprocessor = get_preprocessed_data(data_path)
    
    # Check if we have enough data
    if len(y) < 10:
        raise ValueError("Not enough data for training. Check your dataset.")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Build full pipeline with preprocessing and model
    model = create_model_pipeline(model_type, params)
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    print(f"Training {model_type} model...")
    full_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model performance...")
    metrics = evaluate_model(full_pipeline, X_test, y_test)
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Get feature importance if available
    feature_names = list(X.columns)
    if hasattr(full_pipeline['model'], 'feature_importances_') or hasattr(full_pipeline['model'], 'coef_'):
        get_feature_importance(full_pipeline['model'], feature_names)
    
    # Save the model
    model_path = save_model(full_pipeline, output_dir, filename="loan_approval_model", add_timestamp=True)
    
    return full_pipeline, metrics, model_path

if __name__ == "__main__":
    # Example usage
    data_path = "loan_data.csv"
    output_dir = "models"
    
    # Train and evaluate a random forest model
    model, metrics, model_path = train_and_evaluate(
        data_path=data_path,
        output_dir=output_dir,
        model_type='random_forest',
        params={'n_estimators': 100, 'max_depth': 10}
    )

# Loan Approval ETL Pipeline with Airflow and MLflow

This project implements an ETL (Extract, Transform, Load) pipeline for a loan approval prediction system using Apache Airflow and MLflow. The pipeline simulates daily arrival of new data, processes it, trains a machine learning model, and logs the results to MLflow.

## Pipeline Overview

The ETL pipeline consists of the following steps:

1. **Data Ingestion**: Simulates the daily arrival of new loan application data.
2. **Data Preprocessing**: Cleans and transforms the raw data for model training.
3. **Model Training**: Trains a machine learning model on the preprocessed data.
4. **Model Logging**: Logs the model and its performance metrics to MLflow.
5. **User Interface**: Provides a Streamlit web application for users to submit loan applications and receive predictions.

## Architecture

```mermaid
graph LR
    A[New Data] --> B[Load Data]
    B --> C[Preprocess Data]
    C --> D[Train Model]
    D --> E[Log to MLflow]
    E --> F[Updated Model]
    F --> G[Streamlit App]
    G --> H[User Predictions]
    
    style G fill:#ff9900,stroke:#333,stroke-width:2px
    style H fill:#66ccff,stroke:#333,stroke-width:2px
```

## Components

- **Airflow DAG**: Orchestrates the ETL pipeline and schedules it to run daily.
- **Data Preprocessing**: Cleans and transforms the raw data using scikit-learn pipelines.
- **Model Training**: Trains a Random Forest classifier on the preprocessed data.
- **MLflow**: Tracks model parameters, metrics, and artifacts.
- **Streamlit App**: Provides a user-friendly interface for submitting loan applications and getting predictions.

## Setup Instructions

### Prerequisites

- Docker and Docker Compose

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd loan-approval-etl
   ```

2. Build and start the containers:
   ```bash
   cd airflow
   docker-compose up -d
   ```

3. Access the Airflow UI:
   - Open your browser and navigate to `http://localhost:8080`
   - Login with username `airflow` and password `airflow`

4. Access the MLflow UI:
   - Open your browser and navigate to `http://localhost:5000`

5. Access the Streamlit App:
   - Open your browser and navigate to `http://localhost:8501`

### Directory Structure

```
airflow/
├── dags/                  # Airflow DAG definitions
│   └── etl_pipeline.py    # Main ETL pipeline DAG
├── include/               # Python modules used by the DAGs
│   ├── data_preprocessing.py  # Data preprocessing functions
│   └── train_model.py     # Model training functions
├── app/                   # Streamlit application
│   ├── streamlit_app.py   # Loan approval prediction app
│   └── simple_streamlit_app.py  # Simplified rule-based prediction app
├── tests/                 # Test scripts
│   └── test_etl_pipeline.py  # Tests for the ETL pipeline components
├── models/                # Directory for trained models
├── Dockerfile             # Dockerfile for Airflow
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── run.sh                 # Script to run the Airflow services
├── run_tests.sh           # Script to run the tests
├── run_streamlit.sh       # Script to run the Streamlit app
├── run_streamlit_with_sample.sh  # Script to run Streamlit with a sample model
├── run_simple_streamlit.sh  # Script to run the simplified Streamlit app
├── generate_sample_model.py  # Script to generate a sample model
├── init_data_dirs.sh      # Script to initialize data directories
└── .gitignore             # Git ignore file
```

## Usage

1. The ETL pipeline is scheduled to run daily.
2. You can manually trigger the pipeline from the Airflow UI.
3. View model metrics and artifacts in the MLflow UI.
4. Run the ETL pipeline without Docker:
   ```bash
   # Initialize data directories
   ./init_data_dirs.sh
   
   # Run the ETL pipeline directly
   ./run_etl_without_docker.sh
   ```
5. Access the Streamlit app to submit loan applications and get predictions:
   ```bash
   # Run the simplified Streamlit app (rule-based, no model needed)
   ./run_simple_streamlit.sh
   
   # Run locally with a sample model (for demonstration without running the ETL pipeline)
   ./run_streamlit_with_sample.sh
   
   # Run locally (requires the ETL pipeline to have been run at least once)
   ./run_streamlit.sh
   
   # Or access via Docker
   # Open your browser and navigate to http://localhost:8501
   ```
5. Generate a sample model for demonstration purposes:
   ```bash
   python generate_sample_model.py
   ```

## Customization

- **Model Parameters**: Modify the model parameters in the `train_model` task in `etl_pipeline.py`.
- **Data Sources**: Update the `simulate_new_data` function in `etl_pipeline.py` to connect to your actual data source.
- **Preprocessing Steps**: Customize the preprocessing steps in `data_preprocessing.py`.
- **Streamlit UI**: Modify the Streamlit app in `app/streamlit_app.py` to customize the user interface, add new features, or change the prediction explanation logic.

## Monitoring

- **Airflow**: Monitor DAG runs, task status, and logs in the Airflow UI.
- **MLflow**: Track model performance metrics and compare different runs in the MLflow UI.
- **Streamlit**: Monitor user interactions and predictions through the Streamlit app logs.

## Troubleshooting

- **DAG not visible in Airflow UI**: Check the DAG file for syntax errors.
- **Task failures**: Check the task logs in the Airflow UI for error messages.
- **MLflow connection issues**: Ensure the MLflow server is running and accessible.
- **Streamlit app not loading**: Check if the model files exist in the models directory. Run the ETL pipeline at least once to generate the initial model, use the `run_streamlit_with_sample.sh` script which will automatically generate a sample model, or use the `run_simple_streamlit.sh` script which uses a rule-based approach without requiring a model.
- **Prediction errors in Streamlit**: Verify that the preprocessing steps in the app match those used during model training.

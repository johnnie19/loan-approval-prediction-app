FROM apache/airflow:2.7.1-python3.10

USER root

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        default-libmysqlclient-dev \
        libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements file
COPY requirements.txt /opt/airflow/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Create directories for data and models
RUN mkdir -p /opt/airflow/data/raw \
    && mkdir -p /opt/airflow/data/processed \
    && mkdir -p /opt/airflow/models

# Copy the DAGs
COPY dags/ /opt/airflow/dags/

# Copy the preprocessing and model training modules
COPY include/ /opt/airflow/include/

# Copy the Streamlit app
COPY app/ /opt/airflow/app/

# Set environment variables
ENV PYTHONPATH=/opt/airflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

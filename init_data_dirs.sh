#!/bin/bash

# Script to initialize the data directories for the ETL pipeline

# Define the directories to create
DATA_DIR="data"
RAW_DATA_DIR="${DATA_DIR}/raw"
PROCESSED_DATA_DIR="${DATA_DIR}/processed"
MODELS_DIR="models"
LOGS_DIR="logs"

# Create the directories
echo "Creating data directories..."
mkdir -p "${RAW_DATA_DIR}"
mkdir -p "${PROCESSED_DATA_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${LOGS_DIR}"

# Set permissions
echo "Setting permissions..."
chmod -R 755 "${DATA_DIR}"
chmod -R 755 "${MODELS_DIR}"
chmod -R 755 "${LOGS_DIR}"

echo "Data directories initialized successfully!"
echo "Directory structure:"
echo "- ${RAW_DATA_DIR} (for raw data files)"
echo "- ${PROCESSED_DATA_DIR} (for processed data files)"
echo "- ${MODELS_DIR} (for trained models)"
echo "- ${LOGS_DIR} (for Airflow logs)"

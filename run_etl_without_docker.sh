#!/bin/bash

# Create necessary directories
mkdir -p data/raw data/processed models logs

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the ETL pipeline
python run_etl_pipeline.py

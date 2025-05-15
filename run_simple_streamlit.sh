#!/bin/bash

# Script to run the simplified Streamlit application
# This is useful for demonstration purposes without having to run the full ETL pipeline

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if the models directory exists
if [ ! -d "models" ]; then
    echo "Models directory not found. Creating it..."
    mkdir -p models
fi

# Check if there are any model files
if [ -z "$(ls -A models/*.joblib 2>/dev/null)" ]; then
    echo "No model files found. Generating a sample model..."
    python generate_sample_model.py
fi

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing now..."
    pip install streamlit
fi

# Run the Streamlit app
echo "Starting Streamlit application..."
streamlit run app/simple_streamlit_app.py -- --server.port=8501 --server.address=0.0.0.0

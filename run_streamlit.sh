#!/bin/bash

# Script to run the Streamlit application for loan approval prediction

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing now..."
    pip install streamlit
fi

# Run the Streamlit app
echo "Starting Streamlit application..."
streamlit run app/streamlit_app.py -- --server.port=8501 --server.address=0.0.0.0

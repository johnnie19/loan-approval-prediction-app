#!/bin/bash

# Script to run tests for the ETL pipeline

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create test data directory if it doesn't exist
mkdir -p tests/test_data

# Run the tests
echo "Running ETL pipeline tests..."
python -m unittest discover -s tests

# Check the exit code
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Please check the output above for details."
fi

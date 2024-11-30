#!/bin/bash

# Create a new virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Notify user
echo "Environment setup is complete."

#!/bin/bash

# Create a virtual environment and install dependencies
python -m venv transformer_env
source transformer_env/bin/activate
pip install -r requirements.txt

# Run the training script
python src/trainer.py

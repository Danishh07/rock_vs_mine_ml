#!/bin/bash

# This script will be run during deployment to train the model
echo "🚀 Building Rock vs Mine Prediction Model..."

# Install dependencies
pip install -r requirements.txt

# Run the training script to generate models
python main.py

echo "✅ Build completed successfully!"

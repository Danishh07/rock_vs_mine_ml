#!/bin/bash

# This script will be run during deployment to train the model
echo "🚀 Building Rock vs Mine Prediction Model..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if data file exists
if [ ! -f "data/sonar.csv" ]; then
    echo "❌ Error: data/sonar.csv not found!"
    echo "Please ensure the dataset is included in your repository"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/trained_models
mkdir -p results/plots
mkdir -p results/metrics

# Run the training script to generate models
echo "🤖 Training models..."
python main.py

# Verify models were created
if [ -f "models/trained_models/best_model.joblib" ] && [ -f "models/trained_models/scaler.joblib" ]; then
    echo "✅ Models created successfully!"
    ls -la models/trained_models/
else
    echo "❌ Error: Models were not created properly!"
    echo "Checking directory contents:"
    ls -la models/
    if [ -d "models/trained_models" ]; then
        ls -la models/trained_models/
    fi
    exit 1
fi

echo "✅ Build completed successfully!"

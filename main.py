"""
This script provides a complete pipeline for Rock vs Mine prediction using SONAR data.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_and_preprocess_data
from model_training import train_multiple_models
from model_evaluation import evaluate_models, plot_confusion_matrices

def main():
    """
    Main function to execute the Rock vs Mine prediction pipeline
    """
    print("🚀 Rock vs Mine Prediction Project")
    print("=" * 50)
    
    # Check if data file exists
    data_path = os.path.join("data", "sonar.csv")
    if not os.path.exists(data_path):
        print("❌ Dataset not found!")
        print(f"Please place the SONAR dataset at: {data_path}")
        print("You can download it from: https://www.kaggle.com/datasets/rupakroy/sonarcsv")
        return
    
    try:
        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        
        # Train multiple models
        print("\n🤖 Training multiple models...")
        models = train_multiple_models(X_train, y_train)
        print("✅ Models trained successfully!")
        
        # Evaluate models
        print("\n📈 Evaluating models...")
        results = evaluate_models(models, X_test, y_test)
        
        # Display results
        print("\n🏆 Model Performance Results:")
        print("-" * 40)
        for model_name, metrics in results.items():
            print(f"{model_name:20} | Accuracy: {metrics['accuracy']:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        print(f"\n🥇 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Generate confusion matrices
        print("\n📊 Generating visualizations...")
        plot_confusion_matrices(models, X_test, y_test)
        
        # Save the best model and scaler
        print("\n💾 Saving best model...")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_model = models[best_model_name]
        
        # Create models directory
        os.makedirs("models/trained_models", exist_ok=True)
        
        # Save best model and scaler
        model_path = "models/trained_models/best_model.joblib"
        scaler_path = "models/trained_models/scaler.joblib"
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Best model ({best_model_name}) saved to: {model_path}")
        print(f"✅ Scaler saved to: {scaler_path}")
        
        # Save results
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results.keys()],
            'Precision': [results[model]['precision'] for model in results.keys()],
            'Recall': [results[model]['recall'] for model in results.keys()],
            'F1-Score': [results[model]['f1_score'] for model in results.keys()]
        })
        
        results_path = os.path.join("results", "metrics", "model_performance.csv")
        results_df.to_csv(results_path, index=False)
        print(f"✅ Results saved to: {results_path}")
        
        print("\n🎉 Project completed successfully!")
        print("Check the 'results' folder for visualizations and metrics.")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please check the data file and try again.")

if __name__ == "__main__":
    main()

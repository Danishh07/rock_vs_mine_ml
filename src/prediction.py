"""
Prediction Module for Rock vs Mine Classification
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

class RockMinePredictionSystem:
    """
    A complete prediction system for Rock vs Mine classification
    """
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the prediction system
        
        Args:
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_names = [f'Feature_{i}' for i in range(60)]  # SONAR has 60 features
        
    def load_model(self, model_path=None):
        """
        Load a trained model from file
        
        Args:
            model_path (str): Path to the model file
        """
        if model_path is None:
            model_path = self.model_path
            
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
        
    def load_scaler(self, scaler_path=None):
        """
        Load a fitted scaler from file
        
        Args:
            scaler_path (str): Path to the scaler file
        """
        if scaler_path is None:
            scaler_path = self.scaler_path
            
        if scaler_path is None or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded from: {scaler_path}")
        
    def predict_single(self, features):
        """
        Make prediction for a single sample
        
        Args:
            features (list or array): Feature values for prediction
            
        Returns:
            dict: Prediction result with class and probability
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Ensure features is numpy array
        features = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
            
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)[0]
            probability = {
                'Rock': proba[0],
                'Mine': proba[1]
            }
            
        # Convert prediction to readable format
        prediction_label = 'Mine' if prediction == 1 else 'Rock'
        
        return {
            'prediction': prediction_label,
            'prediction_code': int(prediction),
            'probabilities': probability,
            'confidence': max(probability.values()) if probability else None
        }
        
    def predict_batch(self, features_list):
        """
        Make predictions for multiple samples
        
        Args:
            features_list (list): List of feature arrays
            
        Returns:
            list: List of prediction results
        """
        results = []
        for features in features_list:
            result = self.predict_single(features)
            results.append(result)
        return results
        
    def predict_from_dataframe(self, df):
        """
        Make predictions from a pandas DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with features
            
        Returns:
            pd.DataFrame: DataFrame with predictions added
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Make copy to avoid modifying original
        result_df = df.copy()
        
        # Scale features if scaler is available
        features = df.values
        if self.scaler is not None:
            features = self.scaler.transform(features)
            
        # Make predictions
        predictions = self.model.predict(features)
        prediction_labels = ['Mine' if pred == 1 else 'Rock' for pred in predictions]
        
        # Add probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
            result_df['Rock_Probability'] = probabilities[:, 0]
            result_df['Mine_Probability'] = probabilities[:, 1]
            result_df['Confidence'] = np.max(probabilities, axis=1)
            
        result_df['Prediction'] = prediction_labels
        result_df['Prediction_Code'] = predictions
        
        return result_df
        
    def save_model_and_scaler(self, model, scaler, model_name="best_model"):
        """
        Save model and scaler for future use
        
        Args:
            model: Trained model to save
            scaler: Fitted scaler to save
            model_name (str): Name for the saved files
        """
        models_dir = "models/trained_models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(models_dir, f"{model_name}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Scaler saved to: {scaler_path}")
        
        return model_path, scaler_path

def create_sample_predictions():
    """
    Create sample predictions for demonstration
    """
    # Sample SONAR data (these are fabricated for demonstration)
    sample_data = [
        # Sample 1 - Likely Rock
        [0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601,
         0.3109, 0.2111, 0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273,
         0.3100, 0.2999, 0.5078, 0.4797, 0.5783, 0.5071, 0.4328, 0.5550,
         0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857, 0.1307, 0.2604,
         0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744,
         0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343,
         0.0383, 0.0324, 0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167,
         0.0180, 0.0084, 0.0090, 0.0032],
        
        # Sample 2 - Likely Mine
        [0.0453, 0.0523, 0.0843, 0.0689, 0.1183, 0.2583, 0.2156, 0.3481,
         0.3337, 0.2872, 0.4918, 0.6552, 0.6919, 0.7797, 0.7464, 0.9444,
         1.0000, 0.8874, 0.8024, 0.7818, 0.5212, 0.4052, 0.3957, 0.3914,
         0.3250, 0.3200, 0.3271, 0.2767, 0.4423, 0.2028, 0.3788, 0.2947,
         0.1984, 0.2341, 0.1306, 0.4182, 0.3835, 0.1057, 0.1840, 0.1970,
         0.1674, 0.0583, 0.1401, 0.1628, 0.0621, 0.0203, 0.0530, 0.0742,
         0.0409, 0.0061, 0.0125, 0.0084, 0.0089, 0.0048, 0.0094, 0.0191,
         0.0140, 0.0049, 0.0052, 0.0044]
    ]
    
    return sample_data

def interactive_prediction():
    """
    Interactive prediction function for user input
    """
    print("🎯 Rock vs Mine Interactive Prediction")
    print("=" * 50)
    
    # Initialize prediction system
    predictor = RockMinePredictionSystem()
    
    # Try to load best model
    try:
        model_path = "models/trained_models/best_model.joblib"
        scaler_path = "models/trained_models/best_model_scaler.joblib"
        
        predictor.load_model(model_path)
        predictor.load_scaler(scaler_path)
        
        print("✅ Prediction system ready!")
        
        while True:
            print("\nOptions:")
            print("1. Use sample data")
            print("2. Load data from CSV file")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                # Use sample data
                samples = create_sample_predictions()
                print(f"\nMaking predictions for {len(samples)} samples...")
                
                for i, sample in enumerate(samples, 1):
                    result = predictor.predict_single(sample)
                    print(f"\nSample {i}:")
                    print(f"  Prediction: {result['prediction']}")
                    if result['probabilities']:
                        print(f"  Rock Probability: {result['probabilities']['Rock']:.3f}")
                        print(f"  Mine Probability: {result['probabilities']['Mine']:.3f}")
                        print(f"  Confidence: {result['confidence']:.3f}")
                        
            elif choice == '2':
                # Load from CSV
                file_path = input("Enter CSV file path: ").strip()
                try:
                    df = pd.read_csv(file_path)
                    print(f"Loaded {len(df)} samples from {file_path}")
                    
                    # Make predictions
                    results_df = predictor.predict_from_dataframe(df)
                    
                    # Save results
                    output_path = file_path.replace('.csv', '_predictions.csv')
                    results_df.to_csv(output_path, index=False)
                    print(f"✅ Predictions saved to: {output_path}")
                    
                    # Show summary
                    prediction_counts = results_df['Prediction'].value_counts()
                    print(f"\nPrediction Summary:")
                    for pred, count in prediction_counts.items():
                        print(f"  {pred}: {count}")
                        
                except Exception as e:
                    print(f"❌ Error loading file: {str(e)}")
                    
            elif choice == '3':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"❌ Error loading prediction system: {str(e)}")
        print("Please make sure you have trained models available.")

if __name__ == "__main__":
    interactive_prediction()

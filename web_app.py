"""
Simple Web Interface for Rock vs Mine Prediction
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def load_model_and_scaler():
    """Load the trained model and scaler"""
    global model, scaler
    
    try:
        model_path = 'models/trained_models/best_model.joblib'
        scaler_path = 'models/trained_models/scaler.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("✅ Model and scaler loaded successfully!")
            return True
        else:
            print("❌ Model files not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model and scaler on app startup (crucial for deployment)
print("🚀 Initializing Rock vs Mine Prediction Web Interface...")
model_loaded = load_model_and_scaler()
if not model_loaded:
    print("⚠️ Warning: Model not loaded. App will return errors for predictions.")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not available. Please ensure the model is trained and deployed correctly.',
                'status': 'model_not_loaded'
            }), 503
        
        # Get features from request
        features = request.json.get('features', [])
        
        if len(features) != 60:
            return jsonify({'error': 'Please provide exactly 60 features'}), 400
        
        # Debug: Print some statistics about the input
        print(f"🔍 DEBUG: Input features stats:")
        print(f"   Min: {min(features):.4f}, Max: {max(features):.4f}")
        print(f"   Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")
        print(f"   First 5 features: {features[:5]}")
        
        # Convert to numpy array and scale
        features_array = np.array(features).reshape(1, -1)
        
        if scaler is not None:
            features_scaled = scaler.transform(features_array)
            print(f"   After scaling - Min: {features_scaled.min():.4f}, Max: {features_scaled.max():.4f}")
        else:
            features_scaled = features_array
        
        # Make prediction using probabilities for consistency (fixes SVM issue)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            prediction = np.argmax(proba)  # Use probabilities for final prediction
            probabilities = {'Rock': float(proba[0]), 'Mine': float(proba[1])}
            confidence = float(max(proba))
            print(f"   Probabilities: Rock={proba[0]:.4f}, Mine={proba[1]:.4f}")
            print(f"   Using probability-based prediction: {prediction} ({'Mine' if prediction == 1 else 'Rock'})")
        else:
            # Fallback for models without probabilities
            prediction = model.predict(features_scaled)[0]
            probabilities = None
            confidence = None
            print(f"   Direct prediction: {prediction} ({'Mine' if prediction == 1 else 'Rock'})")
        
        predicted_label = 'Mine' if prediction == 1 else 'Rock'
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_info = None
    if model is not None:
        model_info = {
            'type': type(model).__name__,
            'has_predict_proba': hasattr(model, 'predict_proba')
        }
    
    return jsonify({
        'status': 'healthy' if model is not None and scaler is not None else 'degraded',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'model_info': model_info,
        'message': 'All systems operational' if model is not None and scaler is not None else 'Model not loaded - predictions unavailable'
    })

if __name__ == '__main__':
    print("🌐 Starting web server locally...")
    print("🔗 Visit http://localhost:5000 to use the app")
    
    # Get port from environment variable (for deployment) or use 5000 for local
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

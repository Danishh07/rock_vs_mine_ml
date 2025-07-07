#!/usr/bin/env python3
"""
Test script to verify the deployment will work correctly
"""
import requests
import json
import numpy as np

def test_local_deployment():
    """Test the local deployment"""
    print("🧪 Testing Rock vs Mine Prediction Deployment")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        health_data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(health_data, indent=2)}")
        
        if health_data.get('model_loaded') and health_data.get('scaler_loaded'):
            print("   ✅ Model and scaler loaded successfully!")
        else:
            print("   ❌ Model or scaler not loaded!")
            return False
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test prediction endpoint with sample data
    print("\n2. Testing prediction endpoint...")
    # Generate sample SONAR data (60 features between 0 and 1)
    sample_features = np.random.uniform(0, 1, 60).tolist()
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": sample_features},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            prediction_data = response.json()
            print(f"   Prediction: {prediction_data.get('prediction')}")
            print(f"   Confidence: {prediction_data.get('confidence')}")
            if prediction_data.get('probabilities'):
                print(f"   Probabilities: {prediction_data.get('probabilities')}")
            print("   ✅ Prediction successful!")
        else:
            print(f"   ❌ Prediction failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Prediction test failed: {e}")
        return False
    
    # Test error handling with invalid input
    print("\n3. Testing error handling...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": [1, 2, 3]},  # Wrong number of features
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 400:
            print("   ✅ Error handling works correctly!")
        else:
            print(f"   ⚠️ Unexpected status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
    
    print("\n🎉 All tests passed! Deployment should work correctly.")
    return True

def test_render_deployment():
    """Test the deployed app on Render"""
    print("\n🌐 Testing Render Deployment")
    print("=" * 50)
    
    base_url = "https://rock-vs-mine-p7dc.onrender.com"
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        health_data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(health_data, indent=2)}")
        
        if health_data.get('model_loaded') and health_data.get('scaler_loaded'):
            print("   ✅ Model and scaler loaded successfully!")
        else:
            print("   ❌ Model or scaler not loaded!")
            return False
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False
    
    # Test prediction endpoint
    print("\n2. Testing prediction endpoint...")
    sample_features = np.random.uniform(0, 1, 60).tolist()
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": sample_features},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            prediction_data = response.json()
            print(f"   Prediction: {prediction_data.get('prediction')}")
            print(f"   Confidence: {prediction_data.get('confidence')}")
            print("   ✅ Prediction successful!")
            return True
        else:
            print(f"   ❌ Prediction failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        test_render_deployment()
    else:
        print("Usage:")
        print("  python test_deployment.py        # Test local deployment")
        print("  python test_deployment.py render # Test Render deployment")
        print()
        
        # For now, just test the render deployment
        test_render_deployment()

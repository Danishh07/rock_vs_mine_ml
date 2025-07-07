"""
Model Training Module for Rock vs Mine Prediction
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_logistic_regression(X_train, y_train, optimize=True):
    """
    Train Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize (bool): Whether to perform hyperparameter tuning
        
    Returns:
        sklearn model: Trained Logistic Regression model
    """
    if optimize:
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"  Best LR parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        return lr

def train_svm(X_train, y_train, optimize=True):
    """
    Train Support Vector Machine model
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize (bool): Whether to perform hyperparameter tuning
        
    Returns:
        sklearn model: Trained SVM model
    """
    if optimize:
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        
        svm = SVC(random_state=42, probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"  Best SVM parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        svm = SVC(random_state=42, probability=True)
        svm.fit(X_train, y_train)
        return svm

def train_random_forest(X_train, y_train, optimize=True):
    """
    Train Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize (bool): Whether to perform hyperparameter tuning
        
    Returns:
        sklearn model: Trained Random Forest model
    """
    if optimize:
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"  Best RF parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        return rf

def train_neural_network(X_train, y_train, optimize=True):
    """
    Train Neural Network model
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize (bool): Whether to perform hyperparameter tuning
        
    Returns:
        sklearn model: Trained Neural Network model
    """
    if optimize:
        # Hyperparameter tuning
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        nn = MLPClassifier(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(nn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"  Best NN parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        nn = MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=1000)
        nn.fit(X_train, y_train)
        return nn

def train_xgboost(X_train, y_train, optimize=True):
    """
    Train XGBoost model
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize (bool): Whether to perform hyperparameter tuning
        
    Returns:
        xgboost model: Trained XGBoost model
    """
    if optimize:
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"  Best XGB parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        return xgb_model

def train_multiple_models(X_train, y_train, optimize=False):
    """
    Train multiple models and return them in a dictionary
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize (bool): Whether to perform hyperparameter tuning
        
    Returns:
        dict: Dictionary of trained models
    """
    models = {}
    
    print("🔄 Training Logistic Regression...")
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train, optimize)
    
    print("🔄 Training Random Forest...")
    models['Random Forest'] = train_random_forest(X_train, y_train, optimize)
    
    print("🔄 Training Support Vector Machine...")
    models['Support Vector Machine'] = train_svm(X_train, y_train, optimize)
    
    return models

def save_models(models, save_dir="models/trained_models"):
    """
    Save trained models to disk
    
    Args:
        models (dict): Dictionary of trained models
        save_dir (str): Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, model in models.items():
        filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        filepath = os.path.join(save_dir, filename)
        joblib.dump(model, filepath)
        print(f"✅ {model_name} saved to: {filepath}")

def load_models(load_dir="models/trained_models"):
    """
    Load trained models from disk
    
    Args:
        load_dir (str): Directory to load models from
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    
    if not os.path.exists(load_dir):
        print(f"❌ Model directory not found: {load_dir}")
        return models
    
    for filename in os.listdir(load_dir):
        if filename.endswith('.joblib'):
            model_name = filename.replace('_model.joblib', '').replace('_', ' ').title()
            filepath = os.path.join(load_dir, filename)
            models[model_name] = joblib.load(filepath)
            print(f"✅ {model_name} loaded from: {filepath}")
    
    return models

"""
Model Evaluation Module for Rock vs Mine Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import os

def evaluate_single_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a single model and return metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC AUC
    roc_auc = None
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models and return results
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation results for each model
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"📊 Evaluating {model_name}...")
        results[model_name] = evaluate_single_model(model, X_test, y_test, model_name)
    
    return results

def plot_confusion_matrices(models, X_test, y_test, save_path=None):
    """
    Plot confusion matrices for all models
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        save_path (str): Path to save the plot
    """
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, model) in enumerate(models.items()):
        row = idx // cols
        col = idx % cols
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rock', 'Mine'], 
                   yticklabels=['Rock', 'Mine'],
                   ax=axes[row, col])
        axes[row, col].set_title(f'{model_name}\nAccuracy: {accuracy_score(y_test, y_pred):.3f}')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    # Hide unused subplots
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrices saved to: {save_path}")
    else:
        default_path = os.path.join("results", "plots", "confusion_matrices.png")
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrices saved to: {default_path}")
    
    plt.show()

def plot_roc_curves(models, X_test, y_test, save_path=None):
    """
    Plot ROC curves for all models
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Rock vs Mine Prediction')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to: {save_path}")
    else:
        default_path = os.path.join("results", "plots", "roc_curves.png")
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to: {default_path}")
    
    plt.show()

def plot_feature_importance(model, feature_names=None, top_n=20, save_path=None):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): Names of features
        top_n (int): Number of top features to show
        save_path (str): Path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        print("❌ Model doesn't have feature importance information")
        return
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance - Top {top_n}')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to: {save_path}")
    
    plt.show()

def generate_classification_report(models, X_test, y_test, save_path=None):
    """
    Generate detailed classification report for all models
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        save_path (str): Path to save the report
    """
    report_data = []
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Get classification report as dict
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Extract metrics for each class and overall
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                row = {
                    'Model': model_name,
                    'Class': class_name,
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1-score', 0),
                    'Support': metrics.get('support', 0)
                }
                report_data.append(row)
    
    # Create DataFrame
    report_df = pd.DataFrame(report_data)
    
    if save_path:
        report_df.to_csv(save_path, index=False)
        print(f"✅ Classification report saved to: {save_path}")
    else:
        default_path = os.path.join("results", "metrics", "classification_report.csv")
        report_df.to_csv(default_path, index=False)
        print(f"✅ Classification report saved to: {default_path}")
    
    return report_df

def compare_models_visualization(results, save_path=None):
    """
    Create a comprehensive comparison visualization of all models
    
    Args:
        results (dict): Dictionary containing evaluation results
        save_path (str): Path to save the plot
    """
    # Prepare data for plotting
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        
        values = [results[model][metric] for model in models]
        
        bars = axes[row, col].bar(models, values, alpha=0.7)
        axes[row, col].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[row, col].set_ylabel(metric.replace("_", " ").title())
        axes[row, col].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels
        axes[row, col].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison plot saved to: {save_path}")
    else:
        default_path = os.path.join("results", "plots", "model_comparison.png")
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison plot saved to: {default_path}")
    
    plt.show()

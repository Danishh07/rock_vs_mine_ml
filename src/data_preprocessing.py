"""
Data Preprocessing Module for Rock vs Mine Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the SONAR dataset from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(file_path, header=None)
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def explore_data(data):
    """
    Perform basic data exploration
    
    Args:
        data (pd.DataFrame): The dataset to explore
        
    Returns:
        dict: Basic statistics about the dataset
    """
    print("📊 Dataset Information:")
    print(f"Shape: {data.shape}")
    print(f"Features: {data.shape[1] - 1}")
    print(f"Samples: {data.shape[0]}")
    
    # Check class distribution
    class_counts = data.iloc[:, -1].value_counts()
    print(f"\nClass Distribution:")
    for class_label, count in class_counts.items():
        print(f"  {class_label}: {count} ({count/len(data)*100:.1f}%)")
    
    # Check for missing values
    missing_values = data.isnull().sum().sum()
    print(f"\nMissing values: {missing_values}")
    
    return {
        'shape': data.shape,
        'class_distribution': class_counts.to_dict(),
        'missing_values': missing_values
    }

def preprocess_data(data, test_size=0.2, random_state=42):
    """
    Preprocess the data for machine learning
    
    Args:
        data (pd.DataFrame): Raw dataset
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    X = data.iloc[:, :-1].values  # All columns except last
    y = data.iloc[:, -1].values   # Last column (target)
    
    # Convert target labels to binary (0 for Rock, 1 for Mine)
    y = np.where(y == 'R', 0, 1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def plot_data_distribution(data, save_path=None):
    """
    Plot the distribution of features and classes
    
    Args:
        data (pd.DataFrame): The dataset
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Class distribution
    class_counts = data.iloc[:, -1].value_counts()
    axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Class Distribution')
    
    # Feature statistics
    feature_data = data.iloc[:, :-1]
    axes[0, 1].hist(feature_data.mean(axis=1), bins=30, alpha=0.7)
    axes[0, 1].set_title('Distribution of Feature Means')
    axes[0, 1].set_xlabel('Mean Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Correlation heatmap (sample of features)
    sample_features = feature_data.iloc[:, ::10]  # Every 10th feature
    corr_matrix = sample_features.corr()
    sns.heatmap(corr_matrix, ax=axes[1, 0], cmap='coolwarm', center=0)
    axes[1, 0].set_title('Feature Correlation (Sample)')
    
    # Box plot for first few features by class
    melted_data = pd.melt(
        data.iloc[:, :5].assign(Class=data.iloc[:, -1]),
        id_vars=['Class'],
        var_name='Feature',
        value_name='Value'
    )
    sns.boxplot(data=melted_data, x='Feature', y='Value', hue='Class', ax=axes[1, 1])
    axes[1, 1].set_title('Feature Distribution by Class (First 5 Features)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Data distribution plot saved to: {save_path}")
    
    plt.show()

def load_and_preprocess_data(file_path, test_size=0.2, random_state=42, plot=True):
    """
    Complete data loading and preprocessing pipeline
    
    Args:
        file_path (str): Path to the CSV file
        test_size (float): Proportion of test set
        random_state (int): Random seed
        plot (bool): Whether to generate plots
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Load data
    data = load_data(file_path)
    
    # Explore data
    stats = explore_data(data)
    
    # Generate plots if requested
    if plot:
        import os
        plot_path = os.path.join("results", "plots", "data_distribution.png")
        plot_data_distribution(data, plot_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        data, test_size, random_state
    )
    
    return X_train, X_test, y_train, y_test, scaler

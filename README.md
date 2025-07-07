# 🚀 Rock vs Mine Prediction Project

## 📋 Project Overview

This project implements a complete machine learning pipeline for classifying SONAR signals as either **rocks** or **mines** (metal cylinders). The project includes data exploration, multiple ML models, evaluation metrics, and both command-line and web interfaces.

## 🎯 What You'll Learn

- **Data Preprocessing**: Feature scaling, train-test splitting, data exploration
- **Machine Learning Models**: Logistic Regression, SVM, Random Forest, Neural Networks
- **Model Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrices
- **Visualization**: Data distributions, model performance, feature importance
- **Deployment**: Web interface for real-time predictions

## 📁 Project Structure

```
Rock vs Mine Prediction/
├── 📊 data/                          # Dataset folder
│   ├── sonar.csv                     # SONAR dataset
│   └── README_dataset.txt            # Dataset information
├── 📓 notebooks/                     # Jupyter notebooks
│   └── 01_data_exploration.ipynb     # Complete analysis notebook
├── 🐍 src/                           # Python modules
│   ├── data_preprocessing.py         # Data handling functions
│   ├── model_training.py            # ML model training
│   ├── model_evaluation.py          # Model evaluation metrics
│   └── prediction.py                # Prediction utilities
├── 🤖 models/                       # Trained models
│   └── trained_models/              # Saved model files
├── 📈 results/                      # Output results
│   ├── plots/                        # Generated visualizations
│   └── metrics/                      # Performance metrics
├── 🌐 templates/                    # Web interface templates
│   └── index.html                    # Main web page
├── 📄 main.py                       # Main execution script
├── 🔧 setup.py                      # Project setup script
├── 🌐 web_app.py                    # Web application
├── 📋 requirements.txt              # Python dependencies
└── 📖 README.md                     # Project documentation
```

## 🚀 Quick Start Guide

### Step 1: Download the Dataset
1. Visit: https://www.kaggle.com/datasets/rupakroy/sonarcsv
2. Download the `sonar.csv` file
3. Place it in the `data/` folder

### Step 2: Set Up Environment
```powershell
# Install dependencies
python setup.py

# Or manually install
pip install -r requirements.txt
```

### Step 3: Run the Analysis
Choose one of these options:

#### Option A: Complete Analysis (Recommended)
```powershell
python main.py
```

#### Option B: Interactive Jupyter Notebook
```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
```

#### Option C: Web Interface
```powershell
# First run main.py to train models, then:
python web_app.py
# Open http://localhost:5000 in your browser
```

## 📊 What the Project Does

### 1. **Data Exploration** 🔍
- Loads and analyzes the SONAR dataset
- Visualizes class distribution (Rock vs Mine)
- Explores feature relationships and correlations
- Identifies data quality issues

### 2. **Data Preprocessing** 🛠️
- Handles categorical labels (R → 0, M → 1)
- Applies feature scaling (StandardScaler)
- Splits data into training/testing sets
- Maintains class balance through stratification

### 3. **Model Training** 🤖
- **Logistic Regression**: Linear baseline model
- **Support Vector Machine**: Non-linear classification
- **Random Forest**: Ensemble method

### 4. **Model Evaluation** 📈
- Accuracy, Precision, Recall, F1-score
- Confusion matrices for each model
- ROC curves and AUC scores
- Feature importance analysis
- Overfitting detection

### 5. **Prediction System** 🔮
- Command-line prediction interface
- Web-based prediction tool
- Model persistence (save/load)
- Real-time classification

## 📈 Expected Results

The models typically achieve:
- **Logistic Regression**: ~85% accuracy
- **SVM**: ~87% accuracy
- **Random Forest**: ~85% accuracy

## 🌐 Web Interface Features

The web interface (`web_app.py`) provides:
- Interactive input for 60 SONAR features
- Real-time predictions with confidence scores
- Example data for testing (Rock and Mine samples)
- Beautiful, responsive design
- Error handling and validation

## 🎓 Learning Outcomes

After completing this project, you'll understand:

1. **End-to-End ML Pipeline**: From raw data to deployed model
2. **Model Comparison**: How different algorithms perform on the same data
3. **Evaluation Metrics**: What metrics matter and when to use them
4. **Feature Engineering**: How preprocessing affects model performance
5. **Model Deployment**: Creating user-friendly interfaces for ML models
6. **Best Practices**: Code organization, documentation, reproducibility


## 🤝 Contributing

Feel free to:
- Add new models or algorithms
- Improve visualizations
- Enhance the web interface
- Add more evaluation metrics
- Create additional notebooks
"""
Setup Script for Rock vs Mine Prediction Project
Run this script to set up the Python environment and install dependencies
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("🔧 Installing Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_data_file():
    """Check if the SONAR dataset is available"""
    data_path = os.path.join("data", "sonar.csv")
    if os.path.exists(data_path):
        print("✅ SONAR dataset found!")
        return True
    else:
        print("❌ SONAR dataset not found!")
        print("📥 Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/rupakroy/sonarcsv")
        print(f"   and place it at: {data_path}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "models/trained_models",
        "results/plots", 
        "results/metrics"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Rock vs Mine Prediction - Project Setup")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating project directories...")
    create_directories()
    
    # Install packages
    print("\n2. Installing Python packages...")
    packages_installed = install_requirements()
    
    # Check for dataset
    print("\n3. Checking for dataset...")
    dataset_available = check_data_file()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"  ✅ Directories created")
    print(f"  {'✅' if packages_installed else '❌'} Python packages")
    print(f"  {'✅' if dataset_available else '❌'} SONAR dataset")
    
    if packages_installed and dataset_available:
        print("\n🎉 Setup complete! You can now run:")
        print("   python main.py")
        print("   or open the Jupyter notebooks in the 'notebooks' folder")
    elif packages_installed:
        print("\n⚠️ Setup almost complete!")
        print("   Please download the SONAR dataset to proceed.")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")

if __name__ == "__main__":
    main()

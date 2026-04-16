🔐 Ransomware Detection Research Pipeline

A machine learning research pipeline for ransomware detection using ensemble learning, feature engineering, SMOTE balancing, feature selection, and stacked models.

This project evaluates multiple tree-based models and a stacking ensemble to detect ransomware with high accuracy and F1 score.

🚀 Features
Data cleaning & preprocessing
Feature engineering for PE file attributes
SMOTE class balancing
Feature selection using Random Forest
Multiple ML models:
Random Forest
XGBoost
LightGBM
CatBoost
Stacking Ensemble
Cross-validation evaluation
Confusion matrix generation
Classification reports
Model saving (.pkl)
Performance comparison plots
📁 Project Structure
research_results/
│
├── models/
│   ├── RandomForest.pkl
│   ├── XGBoost.pkl
│   ├── LightGBM.pkl
│   ├── CatBoost.pkl
│   └── Stacking.pkl
│
├── plots/
│   ├── RandomForest_cm.png
│   ├── XGBoost_cm.png
│   ├── comparison_plot.png
│
├── tables/
│   ├── RandomForest_report.txt
│   ├── XGBoost_report.txt
│   └── model_comparison.csv
📊 Dataset Requirements

Your dataset must contain:

Target column: Benign
Example feature columns:
BitcoinAddresses
ResourceSize
SizeOfStackReserve
NumberOfSections
IatVRA
ExportSize
FileName (optional)
md5Hash (optional)

The pipeline automatically:

removes duplicates
fills missing values
drops filename/hash columns
creates engineered features
🧠 Feature Engineering

The model automatically creates:

has_crypto
size_ratio
section_ratio
high_iat
entropy_like

These features improve ransomware detection performance.

🤖 Models Used
Base Models
RandomForestClassifier
XGBClassifier
LGBMClassifier
CatBoostClassifier
Final Model

Stacking Classifier using:

Random Forest
XGBoost
LightGBM
CatBoost

Meta learner:

XGBoost
📈 Evaluation Metrics

The pipeline computes:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
Cross Validation Score
⚙️ Installation

Install dependencies:

pip install numpy pandas scikit-learn matplotlib joblib
pip install xgboost lightgbm catboost
pip install imbalanced-learn
▶️ How to Run

Place dataset:

data_file (3).csv

Then run:

python rsw.py
📊 Output

After execution:

Models saved to
research_results/models/
Confusion matrices
research_results/plots/
Classification reports
research_results/tables/
Model comparison
model_comparison.csv
comparison_plot.png
🔁 Pipeline Flow
Load Data
   ↓
Clean Dataset
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
SMOTE Balancing
   ↓
Feature Selection
   ↓
Train Models
   ↓
Stacking Ensemble
   ↓
Evaluation
   ↓
Save Models + Results
🎯 Research Goal

To build a high-accuracy ransomware detection system using:

Ensemble learning
Advanced feature engineering
Model stacking
Imbalanced data handling
📌 Reproducibility

Random seed fixed:

SEED = 42

Ensures consistent results across runs.  

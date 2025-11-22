# 🔬 Network Traffic ML Classification Pipeline

This project implements a comprehensive machine learning pipeline for network traffic analysis.  
It includes preprocessing, balancing, model training, evaluation, and visualization.

---

## 📂 Features

- Automatic dataset cleaning (drop unused columns)
- Label encoding and feature scaling
- Handling class imbalance (SMOTE / undersampling)
- Multiple ML models:
  - Random Forest
  - Gradient Boosting (including HistGradientBoosting)
  - XGBoost, LightGBM, CatBoost
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Neural Network (MLP)
  - Logistic Regression, Naive Bayes, LDA, QDA, AdaBoost, Extra Trees
- Hyperparameter search for Random Forest
- Ensemble Voting Classifier
- Performance evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix visualization
- Graphical Intention Graph (network of attack stages)

---

## 📁 Dataset

- CSV format expected
- Must contain a `class` column (used as target)
- Additional network traffic features are used as predictors
- Public dataset available at: [https://github.com/stephane65133/dataset](https://github.com/stephane65133/dataset)

---

## 🚀 Installation

```bash
git clone https://github.com/stephane65133/dataset.git
cd dataset
pip install -r requirements.txt

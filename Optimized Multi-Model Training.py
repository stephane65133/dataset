# === Optimized Multi-Model Training Pipeline ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_score, recall_score
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib

# === 1. Load Data ===
df = pd.read_csv("C:/Users/sir-tech/Desktop/important/stephane/random_data.csv")

# === 2. Clean Columns ===
cols_to_drop = ['tcp.flags.fin', 'tcp.flags.syn', 'tcp.flags.reset', 'tcp.flags.push', 'tcp.flags.ack',
                'tcp.flags.urg', 'ip.flags.df', 'Nop', 'wscale', 'wscale_multiplier',
                'sack_permited', 'maximum_segment_size', 'frame.encap.type', 'sll.halen']
df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# === 3. Process Labels ===
df_cleaned['label'] = df_cleaned['class'].astype(str).str.lower()
df_cleaned = df_cleaned.drop(columns=['class'])

# === 4. Keep Top 10 Classes ===
top_classes = df_cleaned['label'].value_counts().nlargest(10).index
df_filtered = df_cleaned[df_cleaned['label'].isin(top_classes)]

# === 5. Balance Classes ===
min_class_size = df_filtered['label'].value_counts().min()
balanced_df = pd.concat([
    df_filtered[df_filtered['label'] == label].sample(min_class_size, random_state=42)
    for label in top_classes
]).sample(frac=1, random_state=42)

# === 6. Prepare Data ===
X = balanced_df.drop(columns=['label'])
y = balanced_df['label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 7. Feature Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 8. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# === 9. Random Forest Hyperparameter Search ===
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=10, cv=3,
                                   scoring='f1_weighted', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_

# === 10. Model Training ===
models = {
    "Random Forest": best_rf,
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "LightGBM": lgb.LGBMClassifier(),
    "CatBoost": cb.CatBoostClassifier(verbose=0),
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True),
    "HistGB": HistGradientBoostingClassifier(max_iter=100),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Extra Trees": ExtraTreesClassifier()
}

results = []

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })

        # Plot Confusion Matrix
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(f"{name.lower().replace(' ', '_')}_confmat.png")
        plt.close()

    except Exception as e:
        print(f"{name} failed: {e}")

# === 11. Results Summary ===
df_results = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print("\n=== Summary of Model Performances ===")
print(df_results)

# === 12. Ensemble Voting Classifier ===
best_models = sorted(results, key=lambda x: x['F1 Score'], reverse=True)[:3]
ensemble = VotingClassifier(estimators=[
    (bm['Model'], models[bm['Model']]) for bm in best_models
], voting='soft')

ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)

print("\n=== Ensemble VotingClassifier ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_ens):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_ens, average='weighted'):.4f}")

joblib.dump(ensemble, 'ensemble_best_model.pkl')

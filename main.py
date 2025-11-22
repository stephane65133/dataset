import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# ===============================
# 1. Load Dataset
# ===============================
file_path = "C:/Users/sir-tech/Desktop/important/stephane/random_data.csv"
df = pd.read_csv(file_path)

# ===============================
# 2. Drop unnecessary columns
# ===============================
cols_to_drop = [
    'tcp.flags.fin', 'tcp.flags.syn', 'tcp.flags.reset', 'tcp.flags.push', 'tcp.flags.ack',
    'tcp.flags.urg', 'ip.flags.df', 'Nop', 'wscale', 'wscale_multiplier',
    'sack_permited', 'maximum_segment_size', 'frame.encap.type', 'sll.halen'
]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# ===============================
# 3. Encode Target
# ===============================
df['label'] = df['class'].astype(str).str.lower()
df = df.drop(columns=['class'])
top_classes = df['label'].value_counts().nlargest(10).index
df = df[df['label'].isin(top_classes)]
min_class_size = df['label'].value_counts().min()
df = pd.concat([df[df['label'] == label].sample(min_class_size, random_state=42) for label in top_classes]).sample(frac=1, random_state=42)

X = df.drop(columns=['label'])
y = df['label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ===============================
# 4. Random Forest Hyperparameter Search
# ===============================
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=10, cv=3,
                               scoring='f1_weighted', n_jobs=-1, random_state=42)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

# ===============================
# 5. Define Models
# ===============================
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

# ===============================
# 6. Train and Evaluate
# ===============================
results = []
for name, model in models.items():
    try:
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })

        # Confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,7))
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

# ===============================
# 7. Summary
# ===============================
df_results = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print("\n=== Summary of Model Performances ===")
print(df_results)

# ===============================
# 8. Ensemble Voting Classifier
# ===============================
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

# ===============================
# 9. Generate Minimal Intention Graph
# ===============================
G = nx.DiGraph()
intentions = ["Reconnaissance", "Scan_HTTP", "Scan_SSH", "Bruteforce_SSH", "Exploitation_Vuln", "Exfiltration"]
G.add_nodes_from(intentions)

# Example transitions with probabilities
transitions = [
    ("Reconnaissance", "Scan_HTTP", 0.6),
    ("Reconnaissance", "Scan_SSH", 0.4),
    ("Scan_HTTP", "Scan_SSH", 0.7),
    ("Scan_SSH", "Bruteforce_SSH", 0.8),
    ("Bruteforce_SSH", "Exploitation_Vuln", 0.5),
    ("Exploitation_Vuln", "Exfiltration", 0.9),
    ("Bruteforce_SSH", "Exfiltration", 0.3)
]
for u, v, w in transitions:
    G.add_edge(u, v, weight=w)

# Threshold for minimal graph
threshold = 0.5
G_min = nx.DiGraph()
G_min.add_nodes_from(G.nodes())
for u, v, data in G.edges(data=True):
    if data['weight'] >= threshold:
        G_min.add_edge(u, v, weight=data['weight'])

# Plot minimal graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G_min, seed=42)
edge_labels = nx.get_edge_attributes(G_min, 'weight')
nx.draw_networkx_nodes(G_min, pos, node_size=1200, node_color='lightgreen')
nx.draw_networkx_edges(G_min, pos, arrowstyle='->', arrowsize=20, edge_color='black')
nx.draw_networkx_labels(G_min, pos, font_size=10)
nx.draw_networkx_edge_labels(G_min, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=9)
plt.title("Minimal Intention Graph (Threshold = 0.5)")
plt.axis('off')
plt.tight_layout()
plt.savefig("minimal_intention_graph.png")
plt.show()

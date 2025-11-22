import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. Charger les données
file_path = "C:/Users/sir-tech/Desktop/important/stephane/random_data.csv"
df = pd.read_csv(file_path)

# Vérification de la structure du dataset
print("Aperçu du dataset:")
print(df.info())
print(df.head())

# 2. Vérification des colonnes
if "class" not in df.columns:
    raise ValueError("La colonne 'class' est manquante dans le dataset.")

# Séparer les features et la cible
X = df.drop(columns=["class"])
y = df["class"]

# 3. Encoder les colonnes catégoriques si nécessaire
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# 4. Encoder la cible
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# 5. Normaliser les features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 6. Séparer en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)

# 7. Définir les modèles
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

results = {}

# 8. Entraîner et évaluer
for name, model in models.items():
    print(f"Entraînement du modèle {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

# 9. Résultats
for name, metrics in results.items():
    print(f"\nModel: {name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

# 10. Visualisation des matrices de confusion
for name, metrics in results.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=le_y.classes_, yticklabels=le_y.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.show()

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# === 1. Loading the dataset ===
file_path = "C:/Users/sir-tech/Desktop/important/stephane/random_data.csv"
df = pd.read_csv(file_path)

# === 2. Removing useless columns (if they exist) ===
cols_to_drop = [
    'tcp.flags.fin', 'tcp.flags.syn', 'tcp.flags.reset',
    'tcp.flags.push', 'tcp.flags.ack', 'tcp.flags.urg',
    'ip.flags.df', 'Nop', 'wscale', 'wscale_multiplier',
    'sack_permited', 'maximum_segment_size',
    'frame.encap.type', 'sll.halen'
]
df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# === 3. Converting the target into a binary label ===
df_cleaned['attack'] = df_cleaned['class'].apply(lambda x: 0 if 'benign' in str(x).lower() else 1)
df_cleaned = df_cleaned.drop(columns=['class'])

# === 4. Checking class distribution ===
print("Class distribution in the dataset:")
print(df_cleaned['attack'].value_counts())

# === 5. Balancing the classes using undersampling ===
benign = df_cleaned[df_cleaned['attack'] == 0]
attack = df_cleaned[df_cleaned['attack'] == 1]

min_class_size = min(len(benign), len(attack))
if min_class_size == 0:
    raise ValueError("One of the classes is empty after labeling. Check the 'class' column values.")

benign_sampled = benign.sample(min_class_size, random_state=42)
attack_sampled = attack.sample(min_class_size, random_state=42)

df_balanced = pd.concat([benign_sampled, attack_sampled]).sample(frac=1, random_state=42)  # shuffle

# === 6. Splitting features and target ===
X = df_balanced.drop(columns=['attack'])
y = df_balanced['attack']

# === 7. Normalization ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 8. Train-test split + SMOTE ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

# === 9. Defining the models to evaluate ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, early_stopping=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, random_state=42)
}

# === 10. Training and evaluation loop ===
for name, model in models.items():
    print(f"\n🔷 {name} 🔷")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Detailed evaluation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print("📊 Model Evaluation:")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-score  : {f1:.4f}")

        # Full classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        print(cm)

        # Graphical display
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign (0)', 'Attack (1)'],
                    yticklabels=['Benign (0)', 'Attack (1)'])
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error with model {name}: {e}")


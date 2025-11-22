import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # Required for HistGradientBoosting
from sklearn.ensemble import HistGradientBoostingClassifier

# If you want to use SMOTE to balance your dataset, uncomment this part
from imblearn.over_sampling import SMOTE

# Loading the data
file_path = "C:/Users/sir-tech/Desktop/important/stephane/random_data.csv"
df = pd.read_csv(file_path)

# Encoding the target variable
le = LabelEncoder()
y = le.fit_transform(df['class'])

# Selecting the features
X = df.drop(columns=['class'])

# Converting to float32 to save memory
X = X.astype('float32')

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducing dataset size to avoid memory errors
n_samples = 50000
if len(X_scaled) > n_samples:
    X_scaled, _, y, _ = train_test_split(
        X_scaled, y, train_size=n_samples, stratify=y, random_state=42
    )

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# If you want to balance the data with SMOTE, uncomment this part
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

#############################################################
# Training each model

# SVM
try:
    svm = SVC(kernel='linear', probability=True, random_state=42)
    print("Training SVM...")
    svm.fit(X_train, y_train)

    y_pred_svm = svm.predict(X_test)
    print("SVM")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_svm, average='weighted'):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_svm))
except MemoryError:
    print("Insufficient memory to train SVM.")

# KNN
try:
    knn = KNeighborsClassifier(n_neighbors=5)
    print("Training KNN...")
    knn.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    print("KNN")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_knn, average='weighted'):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_knn))
except MemoryError:
    print("Insufficient memory to train KNN.")

# Random Forest
try:
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    print("Training Random Forest...")
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    print("Random Forest")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_rf, average='weighted'):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
except MemoryError:
    print("Insufficient memory to train Random Forest.")

# Neural Network with early stopping to avoid non-convergence
try:
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, early_stopping=True, random_state=42)
    print("Training Neural Network...")
    mlp.fit(X_train, y_train)

    y_pred_mlp = mlp.predict(X_test)
    print("Neural Network")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_mlp, average='weighted'):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_mlp))
except Exception as e:
    print(f"Error while training Neural Network: {e}")

# Gradient Boosting (HistGradientBoosting)
try:
    gb = HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=42)
    print("Training Gradient Boosting (HistGradientBoosting)...")
    gb.fit(X_train, y_train)

    y_pred_gb = gb.predict(X_test)
    print("Gradient Boosting")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_gb, average='weighted'):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_gb))
except Exception as e:
    print(f"Error while training Gradient Boosting: {e}")

import os
import cv2
import joblib
import numpy as np
from features import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === Configuration ===
DATA_DIR = 'dataset'  # Folder containing subfolders for each class (e.g., 'organic/', 'recyclable/')
MODEL_PATH = 'waste_classifier.pkl'

# === Load Data and Extract Features ===
X = []
y = []

print("Loading images and extracting features...")

for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    for file in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file)
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Skipped unreadable file {file_path}")
            continue

        features = extract_features(image)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Classifier ===
print("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# === Save the Model ===
joblib.dump(clf, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

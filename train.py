import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.segmentation import watershed
from skimage.filters import sobel
from scipy import ndimage as ndi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define paths
dataset_path = 'dataset'
model_path = 'waste_classifier.joblib'
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def extract_watershed_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate gradient using Sobel filter for watershed
    gradient = sobel(blurred)
    
    # Mark background and foreground for watershed
    markers = np.zeros_like(gray)
    markers[gray < 30] = 1    # Background
    markers[gray > 150] = 2   # Foreground
    
    # Apply watershed algorithm
    segmentation = watershed(gradient, markers)
    
    # Count the number of segments
    unique_segments = len(np.unique(segmentation))
    
    # Extract color features (histogram)
    hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
    
    # Extract texture features (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    
    # Extract HOG features
    hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=False)
    
    # Combine all features
    color_features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
    watershed_feature = np.array([unique_segments])
    
    features = np.concatenate([color_features, lbp_hist, hog_features[:100], watershed_feature])
    
    return features

def load_dataset():
    X = []
    y = []
    
    print("Loading dataset...")
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        print(f"Processing {class_name} images...")
        
        # Get list of image files
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        
        for img_file in tqdm(image_files, desc=class_name):
            img_path = os.path.join(class_dir, img_file)
            try:
                # Load and resize image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Error loading {img_path}")
                    continue
                    
                image = cv2.resize(image, (200, 200))
                
                # Extract features
                features = extract_watershed_features(image)
                
                X.append(features)
                y.append(class_idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(X), np.array(y)

def train_model():
    # Load dataset
    X, y = load_dataset()
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    
    # Train model
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Plot feature importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Features')
    plt.savefig('feature_importance.png')
    
    return model

if __name__ == "__main__":
    train_model() 
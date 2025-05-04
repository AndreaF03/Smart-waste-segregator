import cv2
import numpy as np
import joblib
import os
from skimage.feature import local_binary_pattern

# Load the trained model from an environment variable or default path
MODEL_PATH = os.getenv('MODEL_PATH', 'waste_classifier.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

def extract_features(segmented_img):
    """
    Extract features from the segmented image, including color histograms, 
    texture features (LBP), and basic shape features (contours).
    """
    # Resize the image to a standard size
    resized = cv2.resize(segmented_img, (64, 64))

    # Color Histograms (HSV color space)
    hsv_img = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Texture Features (Local Binary Pattern - LBP)
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype('float')
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize the histogram

    # Shape Features (Contours)
    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = sum(cv2.contourArea(c) for c in contours)  # Total area of all contours

    # Combine features into one vector
    features = np.hstack([hist, lbp_hist, contour_area])

    return features

def classify_waste(segmented_image_path):
    """
    Classify the waste category based on extracted features from the segmented image.
    Returns the predicted waste category (e.g., 'Recyclable', 'Organic').
    """
    # Read the segmented image
    img = cv2.imread(segmented_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {segmented_image_path}")

    # Extract features
    feats = extract_features(img)

    # Predict the waste category
    label = model.predict([feats])[0]
    return label

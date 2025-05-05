import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_features(image):
    """
    Extracts a combination of color, texture, and shape features from an image.
    """

    # Resize to standard size
    resized = cv2.resize(image, (64, 64))

    # --- Color Histogram (HSV) ---
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # --- Local Binary Pattern (LBP) for texture ---
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize

    # --- Contour-based shape feature ---
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = sum(cv2.contourArea(c) for c in contours)

    # --- Combine all features ---
    features = np.hstack([hist, lbp_hist, contour_area])

    return features
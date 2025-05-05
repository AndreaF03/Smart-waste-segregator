import cv2
import numpy as np
import joblib
import os
import base64
from skimage.filters import sobel
from skimage.segmentation import watershed
from train import extract_watershed_features

# Define classes and model path
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
model_path = 'waste_classifier.joblib'

def load_model():
    """Load the trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)

def predict_image(image_path, model=None):
    """Predict waste class for a given image"""
    if model is None:
        model = load_model()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize image
    image = cv2.resize(image, (200, 200))
    
    # Generate segmentation image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = sobel(blurred)
    markers = np.zeros_like(gray)
    markers[gray < 30] = 1    # Background
    markers[gray > 150] = 2   # Foreground
    segmentation = watershed(gradient, markers)
    
    # Create a colored visualization of the segmentation
    segmentation_display = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for segment_id in np.unique(segmentation):
        mask = segmentation == segment_id
        # Assign random color to each segment
        color = np.random.randint(0, 255, 3)
        segmentation_display[mask] = color
    
    # Overlay segmentation on original image
    overlay = cv2.addWeighted(image, 0.7, segmentation_display, 0.3, 0)
    
    # Add contours to the segmentation visualization
    contour_image = overlay.copy()
    # Create another image with just contours on original
    contours_only = image.copy()
    # Convert segmentation to uint8 for contour detection
    segmentation_uint8 = segmentation.astype(np.uint8)
    # Find contours for each segment
    for segment_id in np.unique(segmentation):
        if segment_id == 0:  # Skip background
            continue
        # Create binary mask for this segment
        segment_mask = (segmentation == segment_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on both images
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv2.drawContours(contours_only, contours, -1, (0, 255, 0), 2)
    
    # Save segmentation image to temporary file and convert to base64
    temp_seg_path = os.path.join(os.path.dirname(image_path), "temp_seg.jpg")
    cv2.imwrite(temp_seg_path, contour_image)
    with open(temp_seg_path, 'rb') as img_file:
        segmentation_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    os.remove(temp_seg_path)  # Clean up temporary file
    
    # Save contours-only image to temporary file and convert to base64
    temp_contours_path = os.path.join(os.path.dirname(image_path), "temp_contours.jpg")
    cv2.imwrite(temp_contours_path, contours_only)
    with open(temp_contours_path, 'rb') as img_file:
        contours_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    os.remove(temp_contours_path)  # Clean up temporary file
    
    # Extract features
    features = extract_watershed_features(image)
    
    # Reshape for prediction
    features = features.reshape(1, -1)
    
    # Get prediction
    prediction_idx = model.predict(features)[0]
    prediction_class = classes[prediction_idx]
    
    # Get probabilities
    probabilities = model.predict_proba(features)[0]
    
    # Create results dictionary
    results = {
        'class_name': prediction_class,
        'confidence': float(probabilities[prediction_idx]),
        'probabilities': {classes[i]: float(prob) for i, prob in enumerate(probabilities)},
        'segmentation_image': segmentation_base64,
        'contours_image': contours_base64
    }
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        results = predict_image(image_path)
        print(f"Predicted class: {results['class_name']}")
        print(f"Confidence: {results['confidence']:.4f}")
        print("\nClass probabilities:")
        for class_name, prob in results['probabilities'].items():
            print(f"- {class_name}: {prob:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 
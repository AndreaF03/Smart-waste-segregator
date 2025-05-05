
import os
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)

# Define file upload and allowed extensions
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the directories exist (absolute path)
os.makedirs(os.path.join(os.getcwd(), UPLOAD_FOLDER), exist_ok=True)  # Using absolute path
os.makedirs(os.path.join(os.getcwd(), RESULT_FOLDER), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image format")
    return image

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# Function to generate markers for watershed segmentation
def generate_markers(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    return markers

# Function to apply watershed segmentation
def apply_watershed(original_image, markers):
    markers = cv2.watershed(original_image, markers)
    original_image[markers == -1] = [0, 0, 255]  # Boundary in red
    return markers, original_image

# Route to handle file upload and segmentation
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'message': 'No image part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Ensure uploads folder exists
        upload_folder_path = os.path.join(os.getcwd(), UPLOAD_FOLDER)
        if not os.path.exists(upload_folder_path):
            os.makedirs(upload_folder_path)

        # Save the file
        file.save(image_path)

        # Process the image
        original = load_image(image_path)
        preprocessed = preprocess_image(original)
        markers = generate_markers(preprocessed)
        markers, result = apply_watershed(original.copy(), markers)

        # Save the result
        result_filename = 'segmented_' + filename
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, result)

        return jsonify({'message': 'Image processed successfully', 'result_image': result_filename}), 200
    else:
        return jsonify({'message': 'Invalid file type'}), 400

# Route to serve the processed image
@app.route('/results/<filename>')
def send_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

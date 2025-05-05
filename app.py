import os
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from predict import predict_image, load_model
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once during startup
model = None
try:
    model = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    global model
    
    # Check if model is loaded
    if model is None:
        try:
            model = load_model()
        except Exception as e:
            return jsonify({'error': f"Could not load model: {str(e)}"}), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make prediction
            results = predict_image(filepath, model)
            
            # Read the image and convert to base64 for display
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Return results
            return jsonify({
                'success': True,
                'filename': filename,
                'class': results['class_name'],
                'confidence': results['confidence'],
                'probabilities': results['probabilities'],
                'image': img_data,
                'segmentation_image': results['segmentation_image'],
                'contours_image': results['contours_image']
            })
        
        except Exception as e:
            return jsonify({'error': f"Error during prediction: {str(e)}"}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 
from flask import Flask, request, send_from_directory, jsonify
from segmentation import watershed_segment
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from classifier import classify_waste 

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'backend', 'uploads')
RESULT_FOLDER = os.path.join(os.getcwd(), 'backend', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return 'Flask backend is running.'

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    result_path = os.path.join(RESULT_FOLDER, filename)

    file.save(image_path)

    # Segment the image
    watershed_segment(image_path, result_path)
        
    label = classify_waste(result_path)

    # Return HTML to display the segmented result
    return f'<img src="http://127.0.0.1:5000/results/{filename}" alt="Segmented Result">'


@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

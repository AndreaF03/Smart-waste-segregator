# Waste Segregation Classifier

This project implements an image classification system for waste segregation using the watershed algorithm. It can classify waste items into six categories: cardboard, glass, metal, paper, plastic, and trash.

## Features

- Image classification using watershed segmentation algorithm
- Feature extraction using color histograms, texture patterns, and HOG
- Web interface for uploading and testing images
- Visualization of prediction confidence

## Dataset

The dataset contains images of different waste items organized in 6 classes:
- Cardboard (403 images)
- Glass (501 images)
- Metal (410 images)
- Paper (594 images)
- Plastic (482 images)
- Trash (137 images)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, run:
```
python train.py
```

This will:
1. Load images from the dataset directory
2. Extract features using watershed segmentation and other techniques
3. Train a Random Forest classifier
4. Save the trained model as `waste_classifier.joblib`

### Testing with a Single Image

To classify a single image, run:
```
python predict.py path/to/your/image.jpg
```

### Web Interface

To start the web interface for testing:
```
python app.py
```

Then open a browser and go to http://127.0.0.1:5000

## How It Works

The classification system works as follows:

1. **Preprocessing**: Each image is resized to 200×200 pixels
2. **Feature Extraction**:
   - Watershed segmentation to identify regions
   - Color histograms for color distribution
   - Local Binary Patterns (LBP) for texture information
   - Histogram of Oriented Gradients (HOG) for shape features
3. **Classification**: RandomForest classifier trained on the extracted features
4. **Prediction**: For a new image, the same features are extracted and fed to the model to predict the waste type

## Files

- `train.py`: Script for training the model
- `predict.py`: Script for classifying individual images
- `app.py`: Flask web application
- `templates/index.html`: Web interface template
- `requirements.txt`: Required Python packages

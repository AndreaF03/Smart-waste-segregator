import cv2
import numpy as np
import os

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f" Image not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f" Invalid image format: {image_path}")
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

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

def apply_watershed(original_image, markers):
    markers = cv2.watershed(original_image, markers)
    original_image[markers == -1] = [0, 0, 255]  # Mark boundaries in red
    return markers, original_image

def calculate_iou(pred_mask, gt_mask):
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    return 1.0 if union == 0 else intersection / union

def evaluate_segmentation(segmented, ground_truth):
    _, seg_bin = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)
    _, gt_bin = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    iou = calculate_iou(seg_bin, gt_bin)
    print(f" Mean Intersection over Union (mIoU): {iou:.4f}")
    return iou

if __name__ == "__main__":
    image_path = "D:/project/WatershedSegmentationProject/data/input_image.jpg"
    gt_path = "D:/project/WatershedSegmentationProject/data/ground_truth_mask.png"
    output_path = "D:/project/WatershedSegmentationProject/data/watershed_result.jpg"
    
    try:
        original = load_image(image_path)
        preprocessed = preprocess_image(original)
        markers = generate_markers(preprocessed)
        markers, result = apply_watershed(original.copy(), markers)
        cv2.imwrite(output_path, result)
        print(f" Watershed segmentation result saved to: {output_path}")
        
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                pred_mask = np.uint8((markers > 0) * 255)
                evaluate_segmentation(pred_mask, gt_mask)
            else:
                print(" Ground truth mask is not a valid grayscale image.")
        else:
            print(" Ground truth mask not found, skipping evaluation.")
    except Exception as e:
        print(f" Error: {e}")

import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image format")
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
    original_image[markers == -1] = [0, 0, 255]
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
    print(f"Mean Intersection over Union (mIoU): {iou:.4f}")
    return iou

if __name__ == "__main__":
    image_path = "data/input_image.jpg"
    gt_path = "data/ground_truth_mask.png"  # Update the ground truth path as needed
    output_path = "data/watershed_result.jpg"

    original = load_image(image_path)
    preprocessed = preprocess_image(original)

    markers = generate_markers(preprocessed)
    markers, result = apply_watershed(original.copy(), markers)

    cv2.imwrite(output_path, result)
    print(f"Watershed segmentation result saved to: {output_path}")

    try:
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            cv2.imwrite('data/segmented_output.jpg', result)
            cv2.imwrite('data/ground_truth.jpg', gt_mask)
            print("Segmented and ground truth masks saved for comparison.")

            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, pred_bin = cv2.threshold(result_gray, 127, 255, cv2.THRESH_BINARY)
            _, gt_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
            
            iou = calculate_iou(pred_bin, gt_bin)
            print(f"Adjusted IoU: {iou:.4f}")
        else:
            print("Ground truth mask not found or invalid.")
    except Exception as e:
        print("Evaluation skipped:", e)

    markers_display = np.uint8(markers * 255 / np.max(markers))
    cv2.imshow('Markers', markers_display)
    cv2.waitKey(2000)  # Automatically close after 2 seconds
    cv2.destroyAllWindows()

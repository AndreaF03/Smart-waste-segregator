import cv2
import numpy as np

# Create a blank mask (same size as image)
image = cv2.imread("data/input_image.jpg")
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Draw a filled white circle (simulating an object)
cv2.circle(mask, (150, 150), 50, 255, -1)

# Save mask
cv2.imwrite("data/ground_truth_mask.png", mask)

cv2.imshow("Ground Truth Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

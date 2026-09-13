"""
Program 9: Mean vs Gaussian vs Median
- Applies Mean, Gaussian and Median filters to the SAME noisy input image
- Saves all three results separately
"""

import cv2

# Read the SAME noisy input image for all three filters
img = cv2.imread("input.jpg")

# --- Mean filter ---
MEAN_KERNEL = (5, 5)
mean_result = cv2.blur(img, MEAN_KERNEL)
cv2.imwrite("output_mean.png", mean_result)
print(f"Applied mean filter with kernel {MEAN_KERNEL} -> saved output_mean.png")

# --- Gaussian filter ---
GAUSSIAN_KERNEL = (5, 5)
gaussian_result = cv2.GaussianBlur(img, GAUSSIAN_KERNEL, 0)
cv2.imwrite("output_gaussian.png", gaussian_result)
print(f"Applied Gaussian filter with kernel {GAUSSIAN_KERNEL} -> saved output_gaussian.png")

# --- Median filter (best suited for salt-and-pepper noise) ---
MEDIAN_KERNEL = 5
median_result = cv2.medianBlur(img, MEDIAN_KERNEL)
cv2.imwrite("output_median.png", median_result)
print(f"Applied median filter with kernel size {MEDIAN_KERNEL} -> saved output_median.png")

print("All three filtered outputs saved successfully.")

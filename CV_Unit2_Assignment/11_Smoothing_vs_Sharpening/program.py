"""
Program 11: Smoothing and Sharpening Comparison
- Uses the same input image
- Generates one smoothed output and one sharpened output
- Saves them as output_smooth.png and output_sharp.png
"""

import numpy as np
import cv2

# Read the same input image for both operations
img = cv2.imread("input.jpg")

# --- Smoothing: Gaussian blur ---
SMOOTH_KERNEL = (7, 7)
smoothed = cv2.GaussianBlur(img, SMOOTH_KERNEL, 0)
cv2.imwrite("output_smooth.png", smoothed)
print(f"Applied Gaussian smoothing with kernel {SMOOTH_KERNEL} -> saved output_smooth.png")

# --- Sharpening: custom kernel via filter2D ---
# Same style sharpening kernel as Program 10: boosts the center pixel and
# subtracts the four direct neighbors to enhance edges.
sharpening_kernel = np.array([
    [ 0, -1,  0],
    [-1,  9, -1],
    [ 0, -1,  0]
], dtype=np.float32)
sharpened = cv2.filter2D(img, -1, sharpening_kernel)
cv2.imwrite("output_sharp.png", sharpened)
print("Applied sharpening kernel -> saved output_sharp.png")

print("Smoothing vs Sharpening comparison outputs saved successfully.")

"""
Program 2: Controlled Brightness Enhancement
- Reads a dark image
- Increases brightness by a chosen constant while clipping to valid [0,255] range
- Saves the enhanced image
- Prints a pixel value before and after enhancement
"""

import cv2
import numpy as np

# Read the dark input image
img = cv2.imread("input.jpg")

# Constant brightness increase value (chosen empirically for a visibly dark image)
BRIGHTNESS_CONSTANT = 80

# Pick a sample pixel to compare before/after (row=50, col=50)
sample_row, sample_col = 50, 50
pixel_before = img[sample_row, sample_col].copy()

# cv2.add handles saturation (clipping at 255) correctly for uint8 images,
# unlike plain numpy addition which would wrap around (overflow).
brightness_array = np.full(img.shape, BRIGHTNESS_CONSTANT, dtype=np.uint8)
enhanced = cv2.add(img, brightness_array)

pixel_after = enhanced[sample_row, sample_col]

print("Pixel value BEFORE enhancement at (50,50):", pixel_before)
print("Pixel value AFTER enhancement at (50,50):", pixel_after)

# Save the enhanced image
cv2.imwrite("output.png", enhanced)
print("Saved brightness-enhanced output as output.png")

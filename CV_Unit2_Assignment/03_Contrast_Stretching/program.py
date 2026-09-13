"""
Program 3: Contrast Stretching
- Reads a low-contrast grayscale image
- Finds minimum and maximum intensity values
- Performs linear contrast stretching to expand the intensity range to [0,255]
- Saves the enhanced result
NOTE: This is manual linear contrast stretching, NOT histogram equalization.
"""

import cv2
import numpy as np

# Read image directly in grayscale mode
gray = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Find current minimum and maximum intensity values
min_val = np.min(gray)
max_val = np.max(gray)
print("Original minimum intensity:", min_val)
print("Original maximum intensity:", max_val)

# Linear contrast stretching formula:
# new_pixel = (pixel - min) * (255 / (max - min))
# This maps the narrow [min, max] range to the full [0, 255] range.
stretched = (gray.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
stretched = np.clip(stretched, 0, 255).astype(np.uint8)

print("Stretched minimum intensity:", np.min(stretched))
print("Stretched maximum intensity:", np.max(stretched))

# Save the contrast-stretched result
cv2.imwrite("output.png", stretched)
print("Saved contrast-stretched output as output.png")

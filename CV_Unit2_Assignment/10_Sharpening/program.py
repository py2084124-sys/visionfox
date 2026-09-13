"""
Program 10: Image Sharpening Using a Custom Kernel
- Creates a sharpening kernel
- Applies it using cv2.filter2D()
- Saves the sharpened result as output.png
"""

import numpy as np
import cv2

# Read input image
img = cv2.imread("input.jpg")

# Custom sharpening kernel:
# The center value (9) boosts the current pixel, and the surrounding -1s
# subtract neighboring pixel values, which emphasizes edges/high-frequency
# detail. The kernel sums to 1, so overall image brightness is preserved.
sharpening_kernel = np.array([
    [ 0, -1,  0],
    [-1,  9, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# Apply the custom kernel using 2D convolution/filtering
sharpened = cv2.filter2D(img, -1, sharpening_kernel)

# Save the sharpened result
cv2.imwrite("output.png", sharpened)
print("Sharpening kernel used:\n", sharpening_kernel)
print("Saved sharpened output as output.png")

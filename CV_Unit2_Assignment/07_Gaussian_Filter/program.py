"""
Program 7: Gaussian Smoothing
- Applies Gaussian smoothing to a noisy image
- Saves the smoothed result as output.png
"""

import cv2

# Read the noisy input image
img = cv2.imread("input.jpg")

# Kernel size choice: 5x5 is used because it is large enough to noticeably
# reduce Gaussian noise while still being small enough to preserve edges
# and important image details (a common balanced choice for moderate noise).
# The kernel size must be odd so that there is a well-defined center pixel.
KERNEL_SIZE = (5, 5)
SIGMA = 0  # 0 lets OpenCV auto-calculate sigma from the kernel size

smoothed = cv2.GaussianBlur(img, KERNEL_SIZE, SIGMA)

# Save the Gaussian-smoothed result
cv2.imwrite("output.png", smoothed)
print(f"Applied Gaussian blur with kernel size {KERNEL_SIZE}")
print("Saved Gaussian-smoothed output as output.png")

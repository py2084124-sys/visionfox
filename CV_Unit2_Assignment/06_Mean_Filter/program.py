"""
Program 6: Mean Filtering
- Applies a mean/average filter to an image
- Tested with two different kernel sizes (5x5 and 9x9)
- The final saved output uses the LARGER kernel (9x9), as required
"""

import cv2

# Read the input image (color)
img = cv2.imread("input.jpg")

# --- Test 1: smaller kernel size (5x5) ---
KERNEL_SIZE_SMALL = 5
smoothed_small = cv2.blur(img, (KERNEL_SIZE_SMALL, KERNEL_SIZE_SMALL))
print(f"Applied mean filter with kernel size {KERNEL_SIZE_SMALL}x{KERNEL_SIZE_SMALL}")

# --- Test 2: larger kernel size (9x9) -> this is the required final output ---
KERNEL_SIZE_LARGE = 9
smoothed_large = cv2.blur(img, (KERNEL_SIZE_LARGE, KERNEL_SIZE_LARGE))
print(f"Applied mean filter with kernel size {KERNEL_SIZE_LARGE}x{KERNEL_SIZE_LARGE}")

# Save only the result from the LARGER kernel as the required output
cv2.imwrite("output.png", smoothed_large)
print("Saved mean-filtered (9x9 kernel) output as output.png")

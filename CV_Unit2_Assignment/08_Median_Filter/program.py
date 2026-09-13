"""
Program 8: Salt-and-Pepper Noise Reduction (Median Filtering)
- Uses an image containing salt-and-pepper (impulse) noise
- Applies median filtering
- Saves the filtered result as output.png
"""

import cv2

# Read the input image which visibly contains salt-and-pepper noise
img = cv2.imread("input.jpg")

# Median filtering is highly effective for impulse (salt-and-pepper) noise
# because it replaces each pixel with the median of its neighborhood,
# which removes extreme outlier (0 or 255) pixel values without blurring edges much.
KERNEL_SIZE = 5  # must be odd
filtered = cv2.medianBlur(img, KERNEL_SIZE)

# Save the median-filtered result
cv2.imwrite("output.png", filtered)
print(f"Applied median filter with kernel size {KERNEL_SIZE}")
print("Saved median-filtered output as output.png")

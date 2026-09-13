"""
Program 5: Histogram Equalization with Before/After Comparison
- Performs histogram equalization on a grayscale image
- Saves the equalized image as output.png
- Saves a comparison plot (histogram before vs after) as histogram_comparison.png
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Read image in grayscale
gray = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization using OpenCV's built-in function
equalized = cv2.equalizeHist(gray)

# Save the equalized image
cv2.imwrite("output.png", equalized)
print("Saved equalized image as output.png")

# Calculate histograms before and after equalization for comparison
hist_before = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256]).flatten()

# Create a single comparison plot with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(hist_before, color="blue")
axes[0].set_title("Histogram Before Equalization")
axes[0].set_xlabel("Intensity Value")
axes[0].set_ylabel("Frequency")
axes[0].grid(alpha=0.3)

axes[1].plot(hist_after, color="green")
axes[1].set_title("Histogram After Equalization")
axes[1].set_xlabel("Intensity Value")
axes[1].set_ylabel("Frequency")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("histogram_comparison.png", bbox_inches="tight")
plt.close()
print("Saved histogram comparison plot as histogram_comparison.png")

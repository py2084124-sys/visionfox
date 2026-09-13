"""
Program 4: Histogram Analysis
- Reads a grayscale image
- Calculates and plots its intensity histogram
- Saves the histogram plot as output.png
- Prints the intensity value having the highest frequency
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for saving files without a display
import matplotlib.pyplot as plt

# Read image in grayscale
gray = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Calculate histogram: 256 bins for intensity values 0-255
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

# Find the intensity value with the highest frequency (mode)
peak_intensity = int(np.argmax(hist))
peak_count = int(hist[peak_intensity])
print("Intensity value with highest frequency:", peak_intensity)
print("Frequency (pixel count) at that intensity:", peak_count)

# Plot the histogram
plt.figure(figsize=(8, 5))
plt.plot(hist, color="black")
plt.title("Grayscale Intensity Histogram")
plt.xlabel("Intensity Value (0-255)")
plt.ylabel("Frequency (Pixel Count)")
plt.axvline(peak_intensity, color="red", linestyle="--",
            label=f"Peak = {peak_intensity}")
plt.legend()
plt.grid(alpha=0.3)

# Save the plot (never show, per assignment output-saving rule)
plt.savefig("output.png", bbox_inches="tight")
plt.close()
print("Saved histogram plot as output.png")

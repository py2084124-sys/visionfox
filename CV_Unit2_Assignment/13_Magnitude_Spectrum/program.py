"""
Program 13: Magnitude Spectrum
- Computes the DFT and centered magnitude spectrum of a grayscale image
- Uses log scaling for visualization
- Saves the magnitude spectrum as output.png
"""

import cv2
import numpy as np

# Read grayscale image
gray = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Convert to float32 as required by cv2.dft
float_img = np.float32(gray)

# Compute 2D DFT with complex (real + imaginary) output
dft = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)

# Center the low-frequency component
dft_shifted = np.fft.fftshift(dft)

# Magnitude = sqrt(real^2 + imag^2), computed via cv2.magnitude
magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])

# Log scaling compresses the large dynamic range of magnitude values so
# that both strong low-frequency and weak high-frequency components
# become visible in the saved image.
magnitude_log = np.log(magnitude + 1)

# Normalize to 0-255 range for saving as a viewable image
magnitude_display = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
magnitude_display = magnitude_display.astype(np.uint8)

# Save the magnitude spectrum
cv2.imwrite("output.png", magnitude_display)
print("Magnitude spectrum computed and saved as output.png")

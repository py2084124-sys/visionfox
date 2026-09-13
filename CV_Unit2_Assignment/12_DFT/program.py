"""
Program 12: 2D DFT Computation
- Reads a grayscale image
- Converts it to the required numeric type (float32)
- Computes its 2D DFT using OpenCV
- Shifts the frequency representation so the low-frequency region is centered
- Prints shapes of original image, DFT result and shifted DFT result
"""

import cv2
import numpy as np

# Read image in grayscale
gray = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
print("Original image shape:", gray.shape)

# Convert to float32, required numeric type for cv2.dft
float_img = np.float32(gray)

# Compute the 2D DFT. DFT_COMPLEX_OUTPUT returns a 2-channel array
# (real and imaginary parts) instead of a packed CCS format.
dft = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
print("DFT result shape:", dft.shape)

# Shift the zero-frequency (DC) component from the corners to the center
# of the spectrum, which makes the frequency representation easier to
# interpret and visualize.
dft_shifted = np.fft.fftshift(dft)
print("Shifted DFT result shape:", dft_shifted.shape)

# Save a simple visualization (magnitude, log-scaled) so an output.png exists
magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
magnitude_log = np.log(magnitude + 1)
magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("output.png", magnitude_norm.astype(np.uint8))
print("Saved DFT magnitude visualization as output.png")

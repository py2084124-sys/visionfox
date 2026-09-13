"""
Program 1: Grayscale Conversion and Image Information
- Reads a color image
- Converts it to grayscale
- Saves the grayscale result as output.png
- Prints original shape, grayscale shape, height and width
"""

import cv2

# Read the color image (relative path, works inside this folder)
img = cv2.imread("input.jpg")

# Convert BGR (OpenCV default) image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Print required information
print("Original image shape (H, W, Channels):", img.shape)
print("Grayscale image shape (H, W):", gray.shape)
height, width = gray.shape
print("Height:", height)
print("Width:", width)

# Save the grayscale result
cv2.imwrite("output.png", gray)
print("Saved grayscale output as output.png")

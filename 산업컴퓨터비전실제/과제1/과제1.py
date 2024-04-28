import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Check if image is color (3 channels) or grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # Assuming input is already grayscale

    # Equalize histogram
    equalized_image = cv2.equalizeHist(gray_image)

    # Calculate histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Normalize histogram
    hist_normalized = hist.ravel() / hist.max()

    return equalized_image, hist_normalized

def hsv_value_equalization(image):
    # Check if image is color (3 channels)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split HSV image into channels
        h, s, v = cv2.split(hsv_image)

        # Equalize V channel
        equalized_v = cv2.equalizeHist(v)

        # Merge channels back together
        equalized_hsv_image = cv2.merge([h, s, equalized_v])

        # Convert back to BGR color space
        equalized_color_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2BGR)

        # Calculate histogram
        hist = cv2.calcHist([v], [0], None, [256], [0, 256])

        # Normalize histogram
        hist_normalized = hist.ravel() / hist.max()

        return equalized_color_image, hist_normalized
    else:
        print("Error: Input image does not have 3 channels (BGR).")
        exit()

# Read input image
input_image_path = '../data/Lena.png'  # Replace "input_image.jpg" with your image path
input_image = cv2.imread(input_image_path)

if input_image is None:
    print("Error: Unable to load image.")
    exit()

# Ensure image has 3 channels
if input_image.shape[2] != 3:
    print("Error: Input image does not have 3 channels (BGR).")
    exit()

# Get user input for channel selection
channel = input("Enter the channel to perform histogram equalization (R/G/B): ").upper()

# Perform histogram equalization based on user input
if channel == 'R':
    equalized_image, hist_normalized = histogram_equalization(input_image[:, :, 2])
elif channel == 'G':
    equalized_image, hist_normalized = histogram_equalization(input_image[:, :, 1])
elif channel == 'B':
    equalized_image, hist_normalized = histogram_equalization(input_image[:, :, 0])
else:
    print("Invalid channel selection.")
    exit()

# Display histogram
plt.plot(hist_normalized, color='gray')
plt.xlabel('Intensity')
plt.ylabel('Normalized Frequency')
plt.title('Histogram')
plt.show()

# Display histogram equalized image
cv2.imshow("Histogram Equalized Image", equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perform HSV value equalization
hsv_equalized_image, hist_normalized_hsv = hsv_value_equalization(input_image)

# Display HSV value equalized image
cv2.imshow("HSV Value Equalized Image", hsv_equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

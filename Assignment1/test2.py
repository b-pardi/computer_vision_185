import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your image
image = cv2.imread('horse.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the gradient images Ix and Iy using Sobel operators (you can reuse your previous code)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
Ix = cv2.filter2D(image, -1, sobel_x)
Iy = cv2.filter2D(image, -1, sobel_y)

# Compute the products of derivatives Ixx, Iyy, and Ixy
Ixx = Ix * Ix
Iyy = Iy * Iy
Ixy = Ix * Iy

# Define the standard deviation for Gaussian smoothing
sigma = 1.0  # Adjust as needed

# Define the size of the Gaussian kernel (you can adjust this based on the desired size)
kernel_size = 5  # Example kernel size

# Create a Gaussian kernel using a custom function
def custom_gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - kernel_size // 2)**2 + (y - kernel_size // 2)**2) / (2 * sigma**2)),
        (kernel_size, kernel_size)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel

# Apply Gaussian smoothing to the products of derivatives using the custom Gaussian kernel
Ixx_smoothed = cv2.filter2D(Ixx, -1, custom_gaussian_kernel(kernel_size, sigma))
Iyy_smoothed = cv2.filter2D(Iyy, -1, custom_gaussian_kernel(kernel_size, sigma))
Ixy_smoothed = cv2.filter2D(Ixy, -1, custom_gaussian_kernel(kernel_size, sigma))

# Define a parameter for Harris corner response calculation
k = 0.04  # Adjust as needed

# Compute the Harris matrix H at each pixel
H = (Ixx_smoothed * Iyy_smoothed - Ixy_smoothed**2) - k * (Ixx_smoothed + Iyy_smoothed)**2

# Compute the Harris corner response at each pixel
corner_response = H  # You can further process this if needed

# Display the original image and the computed corner response
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(H, cmap='jet'), plt.title('Harris Matrix')
plt.subplot(133), plt.imshow(corner_response, cmap='jet'), plt.title('Corner Response')
plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

# Gradient operators
G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int8)
G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int8)

# Convolve
grad_x = cv2.filter2D(img, -1, G_x)
grad_y = cv2.filter2D(img, -1, G_y)

magnitude = np.sqrt(grad_x**2 + grad_y**2)  # Magnitude of gradient

# Direction of gradient, arctan2(y, x)
print(grad_x.dtype, grad_y.dtype)
grad_dir = np.arctan2(grad_y, grad_x)

plt.figure(1)
plt.imshow(magnitude, cmap='gray')
plt.savefig("lena_magnitude.png")

plt.figure(2)
plt.imshow(grad_dir, cmap='gray')
plt.savefig("lena_gradient_direction.png")

plt.show()

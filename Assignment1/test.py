import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('horse.jpg')
print(image.shape)

sigma = 2
k_size = 3

def gaussian(x, y):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - k_size//2)**2 + (y - k_size//2)**2) / (2 * sigma**2))

def filter2D(img, k):
    img_height, img_width, channels = img.shape
    filtered_img = np.zeros((img.shape), dtype=float) # init result img
    k_middle = k_size // 2
    
    # iterate through pixels in img
    for c in range(channels):
        for i in range(img_height - k_size + 1):
            for j in range(img_width - k_size + 1):
                k_image = img[i:i+k_size, j:j+k_size, c] # grab section of image using filter
                filtered_img[i+k_middle, j+k_middle, c] = np.sum(k_image * kernel) # assign weighted sum to middle of kernel's img

    return filtered_img


# np.fromfunction() returns matrix based on function inputted
# creates our kernel essentially
kernel = np.fromfunction(gaussian, (k_size, k_size))
kernel /= np.sum(kernel) # normalize kernel
print(kernel)

smoothed = filter2D(image, kernel)
cv2.imshow('Original Image', image)
cv2.imshow('Smoothed Image', smoothed.astype(np.uint8))

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

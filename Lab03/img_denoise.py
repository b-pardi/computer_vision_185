import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(img, mean=100, dev=10):
    gaussian_noise = np.random.normal(mean, dev, (img.shape)) # generate noise
    gaussian_noise = gaussian_noise.astype(img.dtype)
    print(img.dtype, gaussian_noise.dtype)
    noise_img = cv2.add(img, gaussian_noise) # add noise
    noise_img = np.clip(noise_img,0,255) # ensure valid range
    return noise_img


def add_sp_noise(img, salt_probability=0.01, pepper_probability=0.05):
    # generate random matrix [0,1] of img shape
    salt_rand = np.random.rand(img.shape[0], img.shape[1])
    pepper_rand = np.random.rand(img.shape[0], img.shape[1])

    # all pixels rand values below probs become true, else false
    salt_pixels = salt_rand < salt_probability
    pepper_pixels = pepper_rand < pepper_probability

    noise_img = np.copy(img)
    # make True values from above be white/black
    noise_img[salt_pixels] = 255 
    noise_img[pepper_pixels] = 0

    return noise_img

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

img_noise = add_gaussian_noise(img)
img_noise = add_sp_noise(img_noise)

mean_kernel = np.ones((5,5)).astype(np.float64) # mean kernel is array of 1's
mean_kernel /= mean_kernel.shape[0] * mean_kernel.shape[1] # normalize mean kernel

img_denoise_conv = cv2.filter2D(src=img_noise, ddepth=-1, kernel=mean_kernel)
img_denoise_blur = cv2.medianBlur(img_noise,5)

plt.figure(1)
plt.imshow(img_noise,cmap='gray')
plt.savefig("lena_noise.png")

plt.figure(2)
plt.imshow(img_denoise_conv,cmap='gray')
plt.savefig("lena_denoise_conv.png")

plt.figure(3)
plt.imshow(img_denoise_blur,cmap='gray')
plt.savefig("lena_noise_blur.png")
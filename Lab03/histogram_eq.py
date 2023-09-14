import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("bay.png", cv2.IMREAD_GRAYSCALE) # Read gs image
hist, _ = np.histogram(img.flatten(),256,[0,256]) # histogram of pixels in img
cdf = hist.cumsum() # cummulative density function
cdf_norm = cdf / cdf.max()
plt.plot(cdf_norm)
plt.figure(1)
plt.plot(hist)
plt.savefig("histogram_before.png")

eq_img = cv2.equalizeHist(img)
plt.figure(3)
plt.imshow(eq_img,cmap='gray')
plt.savefig("bay_eq.png")
hist, _ = np.histogram(img.flatten(),256,[0,256])
plt.figure(2)
plt.plot(hist)
plt.savefig("histogram_after.png")

plt.show()
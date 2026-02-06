from cmath import rect
import numpy as np
from scipy.signal import find_peaks
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
from scipy import signal
import math


img = cv2.imread('PhotoSets/MASet/Before/MA5.jpeg')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(imggray, 100, 200)

contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
ImgContours = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].imshow(imggray, cmap='gray')
ax[0].set_title('Canny Edges')
ax[0].set_axis_off()

ax[1].imshow(edges, cmap='gray')
ax[1].set_title('Original Image')
ax[1].set_axis_off()

ax[2].imshow(ImgContours)
ax[2].set_title('Detected Contours')
ax[2].set_axis_off()
plt.show()
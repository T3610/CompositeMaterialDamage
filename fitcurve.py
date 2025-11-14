import numpy as np
from scipy.signal import find_peaks
import cv2
from matplotlib import pyplot as plt
import os

import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



img = cv2.imread('corrected\AS1 Front.jpeg', cv2.IMREAD_GRAYSCALE)
size = img.shape
img = img[270:1800,300:1400]
img = cv2.resize(img, (size))
histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
data = []
for index, x in enumerate(histogram):
    data.append([index,histogram[index]]) 
data = np.array(data)
#save data to csv
np.savetxt("undamagedData.csv", data, delimiter=",")
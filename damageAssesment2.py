import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


SCALE = 0.0025 #1mm = 400 pixel

def getDamageInfo(file):
    hasDamage = True
    img = cv2.imread(file)
    #print(img)
    imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    data = getHistogram(imgray,False)
    thresholdValue,thresholedImage = cv2.threshold(imgray,0,255,cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
    damage = data[0:int(thresholdValue)]
    totalDamageArea = sum(damage[:,1])
    if totalDamageArea > 2000*2900*0.9: #if "damages area" is more than 90% of total area --> not damaged
        print(file, ": Not picking up damage")
        thresholdValue,thresholedImage = cv2.threshold(imgray,0,255,cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        hasDamage = False
    return thresholdValue, thresholedImage, hasDamage

    #return totalDamageArea, thresholdValue
    plt.subplot(2, 2, 1)
    plt.bar(colour, freq) # x axis: colour, y axis: freq
    plt.title("Histogram of pixel intensity")
    plt.plot(damage[:,0],damage[:,1],color = 'blue')

    plt.xlabel("Shade of pixel")
    plt.ylabel("Frequency")
    plt.subplot(2,2,2)
    plt.imshow(imgray)
    plt.title("Image of coupon")
    plt.show()
import os

import pandas as pd 

path = "corrected"
files = os.listdir(path)
print(files)
info = []
#info.append(["a","x0","sigma","mean","std dev","area"])

def getHistogram(img,damagedSample):
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    data = []
    for index, x in enumerate(histogram):
        data.append([index,histogram[index]]) 
    data = np.array(data)
    #print(data)
    if damagedSample:
        plt.bar(data[1:,0], data[1:,1]) # x axis: intensity, y axis: freq
    else:
        plt.bar(data[0:,0], data[0:,1])
    plt.title("Histogram of pixel intensity")
    plt.show()
    return data

for file in files:
    imgPath = "corrected/"+file
    #imgPath = "cropped/MA5.jpg"
    data = getDamageInfo(imgPath)
    damageImg = data[1]
    plt.imshow(damageImg)
    plt.show()
    if data[2]: #if sampe has damage
        histData = getHistogram(damageImg,damagedSample=True)
        print(histData)
        np.savetxt("test.csv",histData,delimiter=",")
    else:
        print("Has no existing damage")
#plt.show()
#print((info[0]))
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

SCALE = 0.0025 #1mm = 400 pixel

def Gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def getDamageInfo(file):
    img = cv2.imread(file)
    #print(img)
    imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #print(imgray.shape)
    noPixel = imgray.shape[0]*imgray.shape[1]
    histogram, bin_edges = np.histogram(imgray, bins=256, range=(0, 255))
    freq = []; colour = []; percfreq = []; data = []
    for index, x in enumerate(histogram):
        colour.append(index)
        freq.append(histogram[index])
        data.append([index,histogram[index]])
        percfreq.append((histogram[index]/noPixel)*100) 
    grad = np.gradient(data, axis=0)
    data = np.array(data)
    gradmax = np.argmax(grad, axis = 0)
    gradmin = np.argmin(grad, axis = 0)
    yintercept = ((gradmax + gradmin) // 2)[1] # "max" point
    #print(yintercept)
    #print(grad)
    grad2 = np.gradient(grad[:,1])   
    #print(grad2) 

    for x in range(yintercept,0,-1):
        print(x)
        """if grad2[x] < 200 and grad2[x] > 0:
            UpperCuttOff = x
            break"""
        if grad[x,1] < 500 and grad2[x] > 0:
            UpperCuttOff = x
            break
    lowerCutOff = 0
    for y in range(0,UpperCuttOff):
        if data[y][1] >20:
            lowerCutOff = y
            print(lowerCutOff)
            break

    damage = data[lowerCutOff:UpperCuttOff]
    #print(damage)
    y = damage[:,1]
    x = damage[:,0]
    #print(x)
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    area = sum(y)*0.0025
    #print(mean)
    try:
        popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
    except:
        print("Couldnt fit curve" + "\n" + "Probally due to no damage existing")
        popt = -1
        plt.imshow(imgray)
        plt.title("Image of coupon")
        plt.show()
        #input()
        return -1, -1, -1, -1, 0, 0, lowerCutOff, UpperCuttOff
     #1 mm = 20 pixel so 1 pixel = 0.05mm so 1x1 pixel = 0.0025mm^2
    
    plt.subplot(2, 2, 1)
    plt.bar(colour, freq) # x axis: colour, y axis: freq
    plt.title("Histogram of damage")
    plt.plot(damage[:,0],damage[:,1],color = 'orange')
    plt.xlabel("Frequency of certain shade")
    plt.ylabel("Shade of pixel")
    plt.plot(grad2, color = "green")
    plt.plot(grad[:,1], color = "pink")

    plt.subplot(2,2,2)
    plt.imshow(imgray)
    plt.title("Image of coupon")
    #plt.show()

    
    plt.subplot(2,2,3)
    plt.plot(damage)

    plt.subplot(2,2,4)

    plt.plot(x, y, 'b+:', label='data')
    plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
    plt.legend()
    plt.tight_layout()
    
    plt.show()
    return popt[0],popt[1], popt[2], mean, sigma, area, UpperCuttOff, lowerCutOff #returns [[a,x0,sigma of guassian],total area of damage
    


import os

import pandas as pd 

path = "corrected"
files = os.listdir(path)
print(files)
info = []
#info.append(["a","x0","sigma","mean","std dev","area"])
for file in files:
    imgPath = "corrected/"+file
    #imgPath = "cropped/MA5.jpg"
    info.append(getDamageInfo(imgPath))
#plt.show()
#print((info[0]))
np.savetxt("DamageCurveData.csv",info,delimiter=",")
import cv2
import numpy as np
from matplotlib import pyplot as plt


def furtherCrop(imgPath,iter):
    threshold = 170
    img = cv2.imread(imgPath)
    imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(imgray, threshold, 255,cv2.THRESH_BINARY) # scource, threshold value, max value,type

    contours1, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    size = len(contours1)
    maxContourLength = 0
    maxContourLengthIndex = 0
    for x in range(size):
        print("Contour ", x, " has ", len(contours1[x]), " points.")
        if len(contours1[x]) > maxContourLength:
            maxContourLength = len(contours1[x])
            maxContourLengthIndex = x
    contour = contours1[maxContourLengthIndex]
    image_contours1 = cv2.drawContours(img.copy(), contours1[maxContourLengthIndex], -1, (0,255,0), 10, cv2.LINE_AA)
    """plt.imshow(cv2.cvtColor(image_contours1, cv2.COLOR_BGR2RGB))
    #print(image_contours1.shape)
    plt.title('Contours')
    plt.show()"""

    
    epsilon = 0.005*cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,epsilon,True)
    simpCnt = cv2.drawContours(img, [approx], 0, (255,255,255), 3)
    """plt.imshow(simpCnt)
    plt.show()
    print("simplified contour has",len(approx),"points")"""

    newWidth = 2000
    newHeight = 1.45*newWidth
    print(approx)
    pointForTrans = np.float32([approx[0], approx[1], approx[2], approx[3]])
    correctedPoints = np.float32([[0,0],[newWidth,0],[newWidth,newHeight],[0,newHeight]])
    transMatrix = cv2.getPerspectiveTransform(pointForTrans, correctedPoints)
    result = cv2.warpPerspective(img, transMatrix, (newWidth, int(newHeight)))

    """plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Corrected Perspective')
    plt.show()"""

    imgName = imgPath.split("/")[-1]
    print(imgName)
    newPath = "corrected/"+imgName
    print(imgPath,"----")   
    cv2.imwrite(newPath, result)

import os
path = "cropped"
files = os.listdir(path)
print(files)
for file in files:
    imgPath = "cropped/"+file
    #imgPath = "cropped/MA5.jpg"
    print(imgPath)
    furtherCrop(imgPath,iter=0)
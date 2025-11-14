import numpy as np
from scipy.signal import find_peaks
import cv2
from matplotlib import pyplot as plt
import os


data = np.genfromtxt("test.csv",delimiter=",")
#data = data[1:,:]
#print(data)




# 0 is no damage
# 254 is hole / no material
#255 is area outside coupon
# more layers of delamination the lower the pixel intensity value (shade --> 0)
# AS done, 
# Need ZA5 Back, ZA2 Back
class Sample(object):
    def __init__(self,file,mirrorNeeded):
        self.doSaveImg = True
        self.doShowImg = False
        self.file = file
        self.croppedFilePath = "ImagesNew/" + file
        self.correctedFilePath = "corrected/" + file
        self.filePath = self.correctedFilePath
        print(self.croppedFilePath, "filepath")
        

    def photoCropAndAdjust(self):
        croppedImg = self.furtherCrop(self.croppedFilePath)
        self.correctedImg = self.holeFill(self.correctedFilePath)
        plt.subplot(1, 2, 1)
        plt.title("Image of Sample")
        plt.imshow(self.correctedImg)
        plt.show()



    def GetHist(self):
        self.damageInfo = self.getDamageInfo(self.correctedImg)
        self.damagedSample = self.damageInfo[2]
        self.coupon = self.damageInfo[1]
        #print(self.coupon)
        if self.damagedSample: #if sample has damage
            self.histData = self.createHistogram(self.damageInfo[1],damagedSample=True,plotted=False)
            self.histData = self.histData[1:,:254]
            
            # Find peaks with prominence
            self.peaks, peak_props = find_peaks(self.histData[:,1]) 
            self.troughs, trough_props = find_peaks(-self.histData[:,1])
            #print(peak_props, "peak props")
            self.troughs = np.append([1], self.troughs)
            self.troughs = np.append(self.troughs,[250])
            #print(self.peaks, "peaks", self.troughs, "troughs")
            plt.plot(self.histData)
            plt.plot(self.peaks, self.histData[self.peaks])
            plt.plot(self.troughs, self.histData[self.troughs])
            plt.title("Detected Peaks and Troughs")
            self.pltImgShow()
            
            #print(self.histData)
        else:
            self.histData = None
            self.peaks = [0]
            self.troughs = [0,0]
    
    def holeFill(self,imgPath):
        threshold = 170
        img = cv2.imread(imgPath)
        imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(imgray, threshold, 255,cv2.THRESH_BINARY) # scource, threshold value, max value,type
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_idx, best_contour, best_ellipse, best_score = self.getMostEpllipticalContour(contours, min_points=5, max_area=10000, min_area = 5000)
        if best_contour is not None:
            # Create a mask with the ellipse filled white
            mask = np.zeros(imgray.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, best_ellipse, 255, -1)
            
            # Set all pixels inside ellipse to 255 in the original image
            imgray[mask == 255] = 254
            print(np.shape(imgray))
            #plt.imshow(imgray)
            cont = cv2.drawContours(imgray, [best_contour], 0, (255,255,255), 3)
            if self.doSaveImg:
                plt.imshow(imgray)
                plt.title("Filled Hole using Ellipse Fit")
                self.pltImgShow()
        return imgray
        
    def furtherCrop(self,imgPath):
        
        threshold = 170
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self.doSaveImg:
            plt.imshow(img)
            plt.xlabel("Width (pixels)")
            plt.ylabel("Height (pixels)")
            plt.savefig("reportImg/Before_cropping"+self.file+".jpeg")
            self.pltImgShow()
        #self.pltImgShow()
        ret, thresh = cv2.threshold(imgray, threshold, 255,cv2.THRESH_BINARY) # scource, threshold value, max value,type
        contourInfo = self.getCoupnEdgeContours(img,thresh)
        contoursEdge, maxContourLengthIndex, contour = contourInfo
        image_contours1 = cv2.drawContours(img.copy(), contoursEdge[maxContourLengthIndex], -1, (0,0,255), 30, cv2.LINE_AA)

        if self.doSaveImg:
            plt.imshow(image_contours1)
            plt.savefig("reportImg/ContourPlot"+self.file+".jpeg")
            self.pltImgShow()

        epsilon = 0.005*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        simpCnt = cv2.drawContours(img.copy(), [approx], 0, (255,255,255), 3)
        if self.doSaveImg:
            plt.imshow(simpCnt)
            plt.title("Simplified Contour using Ramer–Douglas–Peucker algorithm")
            self.pltImgShow
        print("simplified contour has",len(approx),"points")

        self.newWidth = 2000
        self.newHeight = 1.45*self.newWidth
        #print(approx)
        pointForTrans = np.float32([approx[0], approx[1], approx[2], approx[3]])
        correctedPoints = np.float32([[0,0],[self.newWidth,0],[self.newWidth,self.newHeight],[0,self.newHeight]])
        transMatrix = cv2.getPerspectiveTransform(pointForTrans, correctedPoints)
        couponResult = cv2.warpPerspective(img, transMatrix, (self.newWidth, int(self.newHeight)))
        
        if self.doSaveImg:
            plt.imshow(couponResult)
            plt.savefig("reportImg/corrected_perspective"+self.file+".jpeg")
            self.pltImgShow()

        imgName = imgPath.split("/")[-1]
        #print(imgName)
        newPath = "corrected/"+imgName
        #
        # print(imgPath,"----")   
        cv2.imwrite(newPath, couponResult)
        return couponResult

    def getMostEpllipticalContour(self, contours, min_points=5, max_area=10000, min_area = 5000):
        """
        From a list of contours return (best_idx, best_contour, ellipse, score)
        ellipse is the ( (cx,cy), (major_axis, minor_axis), angle ) returned by cv2.fitEllipse
        score is IoU between fitted ellipse mask and contour mask (1.0 = perfect match).
        Contours with fewer than min_points or area < max_area are skipped.
        """
        best_idx = None
        best_contour = None
        best_ellipse = None
        best_score = -1.0

        for i, cnt in enumerate(contours):
            if cnt is None or len(cnt) < min_points:
                continue
            area = cv2.contourArea(cnt)
            #print(area, "contour area1", max_area, "max area")
            if int(area) < max_area and int(area) > min_area:
                try:
                    ellipse = cv2.fitEllipse(cnt)  # ((cx,cy),(MA,ma),angle)
                except cv2.error:
                    continue

                # Create small ROI to compare contour mask vs ellipse mask
                x, y, w, h = cv2.boundingRect(cnt)
                if w <= 0 or h <= 0:
                    continue

                mask_contour = np.zeros((h, w), dtype=np.uint8)
                shifted_cnt = cnt.copy()
                shifted_cnt[:, 0, 0] = shifted_cnt[:, 0, 0] - x
                shifted_cnt[:, 0, 1] = shifted_cnt[:, 0, 1] - y
                elipseCnt = cv2.drawContours(mask_contour, [shifted_cnt], -1, 255, -1)
                
                # shift ellipse center into ROI coordinates
                (cx, cy), (MA, ma), angle = ellipse
                cx_shift = cx - x
                cy_shift = cy - y
                axes = (max(1, int(MA/2)), max(1, int(ma/2)))
                mask_ellipse = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(mask_ellipse, (int(round(cx_shift)), int(round(cy_shift))),
                            axes, angle, 0, 360, 255, -1)

                # compute IoU
                inter = cv2.bitwise_and(mask_contour, mask_ellipse)
                union = cv2.bitwise_or(mask_contour, mask_ellipse)
                inter_area = int(np.count_nonzero(inter))
                union_area = int(np.count_nonzero(union))
                if union_area == 0:
                    score = 0.0
                else:
                    score = inter_area / union_area  # 0..1, higher better

                # Optionally penalize extremely elongated ellipses (if desired)
                # aspect = max(MA, ma) / (min(MA, ma) + 1e-6)
                # score = score * (1.0 if aspect < 10 else 0.8) 

                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_contour = cnt
                    best_ellipse = ellipse
            else:
                pass
                #print("Contour ", i, " area ", area, " exceeds max area ", max_area)
                # print(cv2.contourArea(best_contour), "best contour area")
        return best_idx, best_contour, best_ellipse, best_score

    def getCoupnEdgeContours(self,img,thresh):
        contours1, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        size = len(contours1)
        maxContourLength = 0
        maxContourLengthIndex = 0
        for x in range(size):
            #print("Contour ", x, " has ", len(contours1[x]), " points.")
            if len(contours1[x]) > maxContourLength:
                maxContourLength = len(contours1[x])
                maxContourLengthIndex = x
        contour = contours1[maxContourLengthIndex]
        #image_contours1 = cv2.drawContours(img.copy(), contours1[maxContourLengthIndex], -1, (0,255,0), 10, cv2.LINE_AA)
        return contours1, maxContourLengthIndex, contour


    def getDamageInfo(self,imgray):
        hasDamage = True
        plt.subplot(1, 3, 1)
        plt.imshow(imgray)
        mask = (imgray >= 250)
        damageThresholdValue,thresholedImage = cv2.threshold(imgray,0,254,cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU) 
        plt.subplot(1, 3, 2)
        plt.imshow(thresholedImage)
        thresholedImage[mask] = 0
        plt.subplot(1, 3, 3)
        plt.imshow(thresholedImage)
        plt.show()
        """ plt.imshow(thresholedImage)
        plt.title("Thresholded Image for Damage Detection")
        self.pltImgShow()"""
        histData = self.getHistogram(thresholedImage,damagedSample=hasDamage, plotted = True)
        
        damage = histData[0:int(damageThresholdValue)]
        totalDamageArea = sum(damage[:,1])
        if totalDamageArea > self.newHeight*self.newWidth*0.9: #if "damages area" is more than 90% of total area --> not damaged
            print(file, ": Not picking up damage")
            #thresholdValue,thresholedImage = cv2.threshold(imgray,0,254,cv2.THRESH_TOZERO)
            hasDamage = False
        else:
            print(file, ": Damage detected")
        return damageThresholdValue, thresholedImage, hasDamage
    
    def getAllDelaminations(self):
        for x in range(0,len(self.peaks)):
            self.delaminationLayer(x)
            if self.damagedSample == False:
                print("No delaminations to process for undamaged sample:", self.filePath)
                mask = (self.coupon <= 254/2)
                self.coupon[mask] = 0
                mask = (self.coupon > 254/2)
                self.coupon[mask] = 255
                
        self.showImg()


    def delaminationLayer(self,layerOverlapNo):
        if self.damagedSample:
            #print(layerOverlapNo, "layer height")
            #print(self.troughs, "troughs")
            #print(self.peaks, "peaks list")
            self.lowerThres = self.troughs[layerOverlapNo]
            self.upperThres = self.troughs[layerOverlapNo+1]
            #print(self.lowerThres, "LT", self.upperThres, "UT")
            #print(self.peaks[layerOverlapNo], "peak")
            mask = (self.coupon <= self.upperThres) & (self.coupon >= self.lowerThres)
            #print(mask, "mask")
            self.coupon[mask] = self.peaks[layerOverlapNo]
        else:
            self.coupon = self.coupon
            #print("none damaged sample:",self.filePath )

    def showImg(self):
        plt.subplot(2, 2, 1)
        data = self.getHistogram(self.coupon,plotted=False,damagedSample=self.damagedSample)
        plt.bar(data[1:,0], data[1:,1])
        plt.subplot(2, 2, 2)
        plt.imshow(self.coupon)
        if self.doSaveImg:
                print("saving histogram")
                plt.savefig("reportImg/Histogram"+self.file+".jpeg")
        self.pltImgShow()
    

    def createHistogram(self,data,damagedSample=True,plotted=False):
        histogram, bin_edges = np.histogram(data, bins=256, range=(0, 255))
        data = []
        for index, x in enumerate(histogram):
            data.append([index,histogram[index]]) 
        self.data = np.array(data)
        #print(data, "histogram data")
        if plotted:
            if damagedSample:
                
                plt.bar(self.data[1:,0], self.data[1:,1]) # x axis: intensity, y axis: freq
                plt.title("Histogram of pixel intensity (damaged) \n "+self.filePath)
                plt.show()
            else:
                #plt.bar(data[0:,0], data[0:,1])
                plt.bar(self.data[1:,0], self.data[1:,1]) # x axis: intensity, y axis: freq
                plt.title("Histogram of pixel intensity (undamaged) \n"+self.filePath)
                plt.show()
            
        return self.data
    def pltImgShow(self):
        if self.doShowImg:
            plt.show()
        else:
            plt.clf()
        
def plotFrontVsBack(frontData, backData):
    print(frontData)
    plt.bar(frontData[1:255,0], frontData[1:255,1], label="Front Side")
    plt.bar(backData[1:255,0], backData[1:255,1], label="Back Side")
    plt.title("Front vs Back Histogram Comparison")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    #plt.yscale("log")
    plt.legend()
    plt.show()    
    
path = "ImagesNew"
files = os.listdir(path)
print(files)

# files of intrest 
filenames = ["MA1","MA2","MA3","MA4","MA5",
             "AS1","AS2","AS3","AS4","AS5",
             "TB1","TB2","TB3","TB4","TB5",
             "ZA1","ZA2","ZA3","ZA4","ZA5"]
suffix = ["Front","Back"]
filetype = ".jpeg"

coupon = []
print("da")
for file in filenames:
    print(file)
    front = Sample(file+" "+suffix[0]+filetype, mirrorNeeded=False)
    back = Sample(file+" "+suffix[1]+filetype, mirrorNeeded=True)
    coupon = [front, back]
    for item in coupon:
        item.photoCropAndAdjust()
        item.GetHist()
        item.getAllDelaminations()
    Fdata = front.data
    Bdata = back.data
    plotFrontVsBack(Fdata, Bdata)

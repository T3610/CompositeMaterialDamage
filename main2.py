import numpy as np
from scipy.signal import find_peaks
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
from scipy import signal


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
    def __init__(self,filePath):
        self.doSaveImg = True
        self.doShowImg = False
        
        self.ImgFilePath = filePath
        self.correctedFilePath = "corrected/" + filePath

    def photoCropAndAdjust(self):
        
        img = self.furtherCrop(self.ImgFilePath)
        self.correctedImg = self.holeFill(img)
        #self.correctedImg = self.removeGradient(correctedImg)

        correctedImg = self.correctedImg
        """plt.subplot(1, 2, 1)
        plt.title("Image of Sample")
        plt.imshow(self.correctedImg)
        plt.show()"""
        return correctedImg

    def removeGradient(self,img):
        blur = cv2.GaussianBlur(img,(21,21),11)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        corrected = clahe.apply(blur)
        return corrected
    
    def holeFill(self,img):
        threshold = 170
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(imgray, threshold, 255,cv2.THRESH_BINARY) # scource, threshold value, max value,type
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_idx, best_contour, best_ellipse, best_score = self.getMostEpllipticalContour(contours, min_points=5, max_area=10000, min_area = 5000)
        if best_contour is not None:
            # Create a mask with the ellipse filled white
            mask = np.zeros(imgray.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, best_ellipse, 255, -1)
            
            # Set all pixels inside ellipse to 255 in the original image
            imgray[mask == 255] = 254
            #plt.imshow(imgray)
            cont = cv2.drawContours(imgray, [best_contour], 0, (255,255,255), 3)
            
        return imgray
        
    def furtherCrop(self,imgPath):
        
        threshold = 170
        img = cv2.imread(imgPath)
        imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ret, thresh = cv2.threshold(imgray, threshold, 255,cv2.THRESH_BINARY) # scource, threshold value, max value,type
        contourInfo = self.getCoupnEdgeContours(imgray,thresh)
        contoursEdge, maxContourLengthIndex, contour = contourInfo
        image_contours1 = cv2.drawContours(img.copy(), contoursEdge[maxContourLengthIndex], -1, (0,0,255), 30, cv2.LINE_AA)

        epsilon = 0.005*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        simpCnt = cv2.drawContours(img.copy(), [approx], 0, (255,255,255), 3)
        
        pts = approx.reshape(-1, 2)
        tlpts = []
        minDistance = -1
        for index, x in enumerate(pts):
            distance = np.sqrt((pts[index,0]**2)+(pts[index,1]**2))
            if distance < minDistance or minDistance == -1:
                tlpts=x
                minDistance = distance
        
        trpts = []
        imgShape = np.shape(imgray)
        minDistance = -1
        for index, x in enumerate(pts):
            distance = np.sqrt((pts[index,0]-imgShape[0])**2+(pts[index,1])**2)
            if distance < minDistance or minDistance == -1:
                trpts=x
                minDistance = distance
        
        brpts = []
        imgShape = np.shape(imgray)
        minDistance = -1
        for index, x in enumerate(pts):
            distance = np.sqrt((pts[index,0]-imgShape[0])**2+(pts[index,1]-imgShape[1])**2)
            if distance < minDistance or minDistance == -1:
                brpts=x
                minDistance = distance
        
        blpts = []
        imgShape = np.shape(imgray)
        minDistance = -1
        for index, x in enumerate(pts):
            distance = np.sqrt((pts[index,0])**2+(pts[index,1]-imgShape[1])**2)
            if distance < minDistance or minDistance == -1:
                blpts=x
                minDistance = distance
                
        """plt.plot(tlpts[0],tlpts[1],marker="o", color = "blue")
        plt.plot(trpts[0],trpts[1],marker="o",color = "orange")
        plt.plot(brpts[0],brpts[1],marker="o",color = "yellow")
        plt.show()"""
        self.newWidth = 2000
        self.newHeight = 1.45*self.newWidth
        arrangedCorners = [tlpts,trpts,brpts,blpts]
        pointForTrans = np.float32([arrangedCorners[0],arrangedCorners[1],arrangedCorners[2],arrangedCorners[3]])
        correctedPoints = np.float32([[0,0],[self.newWidth,0],[self.newWidth,self.newHeight],[0,self.newHeight]])
        transMatrix = cv2.getPerspectiveTransform(pointForTrans, correctedPoints)
        couponResult = cv2.warpPerspective(img, transMatrix, (self.newWidth, int(self.newHeight)))
        
        
        """plt.imshow(couponResult)
        #plt.savefig("reportImg/corrected_perspective"+self.file+".jpeg")
        plt.show()"""

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

        return best_idx, best_contour, best_ellipse, best_score

    def getCoupnEdgeContours(self,img,thresh):
        contours1, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        size = len(contours1)
        maxContourLength = 0
        maxContourLengthIndex = 0
        for x in range(size):
            if len(contours1[x]) > maxContourLength:
                maxContourLength = len(contours1[x])
                maxContourLengthIndex = x
        contour = contours1[maxContourLengthIndex]
        image_contours1 = cv2.drawContours(img.copy(), contours1[maxContourLengthIndex], -1, (0,255,0), 10, cv2.LINE_AA)
        return contours1, maxContourLengthIndex, contour

    def pltImgShow(self):
        if self.doShowImg:
            plt.show()
        else:
            plt.clf()

class Coupon(object):
    def __init__(self,set,ID):
        filepath = [set+"Set/",set+ID]
        hasDamageList = {"TB1":False,"TB2":False,"TB3":True,"TB4":True,"TB5":False,
                "ZA1":False,"ZA2":False,"ZA3":True,"ZA4":True,"ZA5":False}
        self.hasDamage = hasDamageList[str(set)+str(ID)]
        #input(self.hasDamage)
        before = Sample(filepath[0]+"Before/"+filepath[1]+".jpeg")
        after = Sample(filepath[0]+"After/"+filepath[1]+".jpg")
        self.correctImgBefore = before.photoCropAndAdjust()
        self.correctImgAfter = after.photoCropAndAdjust()

    def getDifference(self,show=False):
        
        self.difference = cv2.subtract(self.correctImgBefore,self.correctImgAfter)
        if show:
            """plt.subplot(1,3,1)
            plt.imshow(self.correctImgBefore)
            plt.subplot(1,3,2)
            plt.imshow(self.correctImgAfter)
            plt.subplot(1,3,3)
            plt.imshow(self.difference)
            plt.show()"""
            pass
        return self.difference
    
    def thresholdImg(self):
        ret,img = cv2.threshold(self.difference,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.mostCentralContour = self.find_most_central_contour(contours)
        self.thresholdImg = img
        """plt.subplot(1,3,1)
        plt.imshow(img)
        
        plt.show()"""

    def damageInfo(self):
        differenceImg = self.difference
        #input(type(differenceImg))
        centralCont = self.mostCentralContour #Cx,Cy
        M = cv2.moments(centralCont[0])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0  # fallback if area is zero
        centralPoint = [cx,cy]

        scatter = []
        #input(scatter)
        for point in centralCont[0]:
            #point = [x,y]
            point = point[0]
            #print(point, centralPoint)
            vectorFromCenter = point-centralPoint
            #print(scatter)
            #input(vectorFromCenter)
            scatter.append(vectorFromCenter)
        scatter = np.asarray(scatter)
        plt.subplot(1,2,1)
        plt.plot(scatter[:,0],scatter[:,1])
        plt.show()
        hgroup = self.get_horizontal_by_y
        print("hgroup",hgroup)
        
    def get_horizontal_by_y(self, contour, tolerance=2):
        pts = contour[:, 0, :]
        horizontal_groups = []

        # Sort by y
        pts_sorted = pts[np.argsort(pts[:, 1])]

        current_group = [pts_sorted[0]]

        for p in pts_sorted[1:]:
            if abs(p[1] - current_group[-1][1]) <= tolerance:
                current_group.append(p)
            else:
                if len(current_group) > 1:
                    horizontal_groups.append(np.array(current_group))
                current_group = [p]

        if len(current_group) > 1:
            horizontal_groups.append(np.array(current_group))

        return horizontal_groups        


    def getAbsorbedEnergy(self,set,ID):

        m = 2.0   # kg
        h = 1.31
        v0 = np.sqrt(2*9.81*h)  # initial velocity (m/s)
        filePath = "ImpactData/"+set+ID+".csv"
        if filePath != "ImpactData/ZA1.csv":
            df = pd.read_csv(filePath, skiprows=8,sep=" ")
            df.columns = ["PointID","Time[ms]","Force[N]","Voltage"]
            df = df[["Time[ms]","Force[N]"]]
            
            
            impulse = np.trapezoid(df["Force[N]"],df["Time[ms]"])
            dv = impulse / (m/1000)
            v_final = v0 + dv
            velocity = v0 + np.cumsum(np.diff(df["Time[ms]"], prepend=0) * df["Force[N]"]/m)
            displacement = np.cumsum(velocity * np.diff(df["Time[ms]"], prepend=0))
            work_done = np.trapezoid(df["Force[N]"], displacement)
            print(work_done)
            #plt.plot(work_done)
            plt.show()
            #input(df.head())

    def find_most_central_contour(self,contours):

        h, w = self.difference.shape
        image_center = np.array([w/2, h/2])
        min_dist = -1
        central_contour = None

        for cnt in contours:
            
            M = cv2.moments(cnt)
            if M["m00"] != 0:  # avoid division by zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = np.array([cx, cy])

                dist = np.linalg.norm(centroid - image_center)
                area = cv2.contourArea(cnt)
                #print(area)
                areaThreshold = 5000
                if dist < min_dist and area > areaThreshold:
                    min_dist = dist
                    central_contour = cnt
                if min_dist == -1:
                    min_dist = dist
                    central_contour = cnt
                    self.centralContourCenter = centroid
            
        #print(central_contour)
        #print("area",cv2.contourArea(central_contour))
        return central_contour, centroid

class Set(object):
    def __init__(self,set):
        photoNumbers = ["1","2","3","4","5"]
        coupons = []
        for photo in photoNumbers:
            coupons.append(Coupon(set,photo))
        for coupon in coupons:
            coupon.getDifference(show=True)
            coupon.thresholdImg()
            coupon.damageInfo()
setNames = ["TB","ZA","MA","AS",]
setNames = ["TB","ZA"]
setList = []
for set in setNames:
    setList.append(Set(set)) #in a set is 5 coupons, in a coupon is a before and after and difference
import numpy as np
from scipy.signal import find_peaks
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
from scipy import signal
import math

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

    def photoCropAndAdjust(self):
        
        img = self.furtherCrop(self.ImgFilePath)
        self.correctedImg = self.holeFill(img)
        #self.correctedImg = self.removeGradient(correctedImg)

        correctedImg = self.correctedImg
        """plt.clf()
        plt.subplot(1, 2, 1)
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
        
        threshold = 175
        img = cv2.imread(imgPath)

        imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # plt.imshow(imgray)
        # plt.show()
        ret, thresh = cv2.threshold(imgray, threshold, 255,cv2.THRESH_BINARY) # scource, threshold value, max value,type
        # plt.imshow(thresh)
        # plt.show()
        contour = self.getCoupnEdgeContours(imgray,thresh)
        
        image_contours1 = cv2.drawContours(imgray.copy(), contour, -1, (0,0,255), 30, cv2.LINE_AA)
        # plt.imshow(image_contours1)
        # plt.show()
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
        return contour

    def pltImgShow(self):
        self.doShowImg = False
        if self.doShowImg:
            plt.show()
        else:
            plt.clf()

class Coupon(object):
    def __init__(self,set,ID):
        self.sample = set+ID
        filepath = [set+"Set/",set+ID]
        hasExistingDamageList = {"TB1":False,"TB2":False,"TB3":True,"TB4":True,"TB5":False,
                "ZA1":False,"ZA2":False,"ZA3":True,"ZA4":True,"ZA5":False,
                "AS1":False,"AS2":False,"AS3":False,"AS4":True,"AS5":True,
                "MA1":False,"MA2":True,"MA3":True,"MA4":True,"MA5":True}
        self.hasExistingDamage = hasExistingDamageList[str(set)+str(ID)]
        #input(self.hasDamage)
        before = Sample(filepath[0]+"Before/"+filepath[1]+".jpeg")
        after = Sample(filepath[0]+"After/"+filepath[1]+".jpg")
        self.correctImgBefore = before.photoCropAndAdjust()
        self.correctImgAfter = after.photoCropAndAdjust()

    def getDifference(self,show=False):
        
        self.difference = cv2.subtract(self.correctImgBefore,self.correctImgAfter)
        show = False
        if show:
            plt.subplot(1,3,1)
            plt.imshow(self.correctImgBefore)
            plt.subplot(1,3,2)
            plt.imshow(self.correctImgAfter)
            plt.subplot(1,3,3)
            plt.imshow(self.difference)
            plt.show()
            pass
        return self.difference
    
    def thresholdImg(self):
        if self.sample == "TB3":
            #self.difference = cv2.GaussianBlur(self.difference,(5,5),5)
            ret, img = cv2.threshold(self.difference,50,255,cv2.THRESH_TOZERO)

        elif self.sample == "TB4":
            ret, img = cv2.threshold(self.difference,55,255,cv2.THRESH_TOZERO)
        else:
            ret,img = cv2.threshold(self.difference,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
        self.find_most_central_contour(img)
    
        self.mostCentralContour = self.central_contour
        area = cv2.contourArea(self.mostCentralContour)
        contouredImage = cv2.drawContours(self.correctImgAfter.copy(), [self.mostCentralContour], 0, (255,255,255), 20)
        self.thresholdedImg = img   
        self.contouredImage = contouredImage

    def damageInfo(self):
        differenceImg = self.difference
        
        centralCont = self.mostCentralContour
        self.damageArea = cv2.contourArea(centralCont)
        M = cv2.moments(centralCont)
        self.center = M
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0  # fallback if area is zero
        centralPoint = [cx,cy]
        self.centralPoint = centralPoint
        scatter = []
        #input(scatter)
        for point in centralCont:
            #point = [x,y]
            point = point[0]
            #print(point, centralPoint)
            vectorFromCenter = point-centralPoint
            #print(scatter)
            #input(vectorFromCenter)
            scatter.append(vectorFromCenter)
        scatter = np.asarray(scatter)
        self.scatter = scatter

        """plt.subplot(1,2,1)
        plt.plot(scatter[:,0],scatter[:,1])
        plt.show()"""
        
        #y axis
        TL = []
        TR = []
        BL = []
        BR = []
        
        for pts in scatter:
            #input(pts)
            if pts[0] <= 0 and pts[1] > 0:
                TL.append(pts)
            elif pts[0] <= 0 and pts[1] <= 0:
                BL.append(pts)
            elif pts[0] > 0 and pts[1] >= 0:
                TR.append(pts)
            else:
                BR.append(pts)
        
        TL = np.asarray(TL)
        TR = np.asarray(TR)
        BL = np.asarray(BL)
        BR = np.asarray(BR)
        N = 11

    
        width = {"TL":min(TL[:,0]),"TR":max(TR[:,0]),
                 "BL":min(BL[:,0]),"BR":max(BR[:,0])}
        truewidth = abs(np.min(self.mostCentralContour[:, :, 0]) - np.max(self.mostCentralContour[:, :, 0]))

        height = {"TL":max(TL[:,1]),"TR":max(TR[:,1]),
                 "BL":min(BL[:,1]),"BR":min(BR[:,1])}
        trueheight = abs(np.min(self.mostCentralContour[:, :, 1]) - np.max(self.mostCentralContour[:, :, 1]))

        self.majorDim = {"Damage Width":truewidth,"Damage Height":trueheight}
        #print(self.majorDim)
        nearestPointTL = min(TL, key=lambda p: math.hypot(p[0], p[1]))
        nearestPointTR = min(TR, key=lambda p: math.hypot(p[0], p[1]))
        nearestPointBL = min(BL, key=lambda p: math.hypot(p[0], p[1]))
        nearestPointBR = min(BR, key=lambda p: math.hypot(p[0], p[1]))
        nearestPoints = [nearestPointTL,nearestPointTR,nearestPointBL,nearestPointBR]
        """
        plt.subplot(2,2,1)
        plt.title("TL:"+str(height["TL"]))
        plt.plot(TL[:,0],TL[:,1])
        plt.plot(nearestPointTL[0],nearestPointTL[1],'ro')
        plt.subplot(2,2,2)
        plt.title("TR"+str(height["TR"]))
        plt.plot(TR[:,0],TR[:,1])
        plt.plot(nearestPointTR[0],nearestPointTR[1],'ro')

        plt.subplot(2,2,3)
        plt.title("BL"+str(height["BL"]))
        plt.plot(BL[:,0],BL[:,1])
        plt.plot(nearestPointBL[0],nearestPointBL[1],'ro')

        plt.subplot(2,2,4)
        plt.title("BR"+str(height["BR"]))
        plt.plot(BR[:,0],BR[:,1])
        plt.plot(nearestPointBR[0],nearestPointBR[1],'ro')

        plt.show()
        """
        extraData = self.extraCrossData([TL,TR,BL,BR],nearestPoints)

        return self.sample, truewidth/20, trueheight/20, self.damageArea*(1/400), self.hasExistingDamage, extraData[0][0]/20, extraData[0][1]/20, extraData[1][0]/20, extraData[1][1]/20

    def extraCrossData(self,data,nearestPoints):
        armLength = []
        armWidth = []
        for ix in range(0,len(data)):
            roundData = data[ix]
            roundData = sorted(roundData, key=lambda x:x[1])
            nearest = nearestPoints[ix]
            nearestFound = False
            ydata = []
            xdata = []
            for x in roundData:
                #print("x", roundData)
                if nearestFound:
                    ydata.append(x)
                    
                else:
                    xdata.append(x)
                if x[0] == nearest[0] and x[1]== nearest[1]:
                    nearestFound=True 

            ydata = np.asarray(ydata)
            xdata = np.asarray(xdata)
            """plt.plot(ydata[:,0],ydata[:,1],color = "red")
            plt.plot(xdata[:,0],xdata[:,1], color = "blue")
            plt.show()"""
            armLength.append(np.ptp(ydata[:,1])) # ,np.ptp(xdata[:,0])
            armWidth.append(np.ptp(xdata[:,0])) # ,np.ptp(xdata[:,1])]
        #print(armLength)
       
        averageArmLength = [(armLength[0]+armLength[1])/2,(armLength[2]+armLength[3])/2]
        averageArmWidth = [(armWidth[0]+armWidth[1])/2,(armWidth[2]+armWidth[3])/2]
        # Varible names not great look at excel spreadsheet for more info
        return averageArmLength,averageArmWidth

    def find_most_central_contour(self,img):
    
    
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntImg = cv2.drawContours(img.copy(), contours, -1, (255,0,0), 3)

        colours = plt.cm.tab20(np.linspace(0, 1, len(contours)))
        #input(colours)
        for i, cnt in enumerate(contours):
            # Ensure shape is always (N, 2)
            cnt = cnt.reshape(-1, 2)

            x = cnt[:, 0]
            y = cnt[:, 1]

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
                
                #print(area)
            
                if dist < min_dist and areaThreshold <= area:
                    min_dist = dist
                    self.central_contour = cnt
                if min_dist == -1:
                    min_dist = dist
                    self.central_contour = cnt
                    centralContourCenter = centroid
            
        #print(central_contour)
        #print("area",cv2.contourArea(central_contour))
        
    def plots(self,data):

        # Plot of before, after and the contoured damage
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True)
        ax[0].imshow(self.correctImgBefore)
        ax[0].set_title("Before Impact")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(self.correctImgAfter)
        ax[1].set_title("After Impact")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].imshow(self.contouredImage)
        ax[2].set_title("Detected Damage")
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        #plt.show()
        plt.tight_layout()
        plt.savefig("InterrimReport/ImgProccess/"+self.sample+".jpeg",dpi=300)
        plt.show()
        plt.clf()
        #Plot of damage scatter

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        plt.imshow(self.contouredImage)
        plt.plot(self.centralPoint[0],self.centralPoint[1],'x',label="Calculated Center of Impact",color="red")
        x_min = np.min(self.mostCentralContour[:, :, 0])
        x_max = np.max(self.mostCentralContour[:, :, 0])
        y_min = np.min(self.mostCentralContour[:, :, 1])
        y_max = np.max(self.mostCentralContour[:, :, 1])

        ax.axvline(x=x_min, color='blue', linestyle='--', label='Damage Width Extremes')
        ax.axvline(x=x_max, color='blue', linestyle='--')
        ax.annotate(
            "", xy=(x_min, 50), xytext=(x_max, 50),
            arrowprops=dict(arrowstyle='<->', color='black')
        )
        plt.text(
            (x_min + x_max) / 2, 250,       # Position (middle of X, slightly above arrow)
            r'$D_w$',    # The label
            ha='center', va='bottom',       # Center the text
            fontsize=10, 
        )

        ax.axhline(y=y_min, color='blue', linestyle='--', label='Damage Height Extremes')
        ax.axhline(y=y_max, color='blue', linestyle='--')
        ax.annotate(
            "", xy=(1800,y_min), xytext=(1800,y_max),
            arrowprops=dict(arrowstyle='<->', color='black')
        )
        ax.text(
            1900,  1500,       # Position (middle of X, slightly above arrow)
            r'$D_h$',    # The label
            ha='center', va='bottom',       # Center the text
            fontsize=10
        )


        textstr = '\n'.join((
        r'$D_w = %.2f$' % (data[1],),
        r'$D_h = %.2f$' % (data[2], ),
        "X = Center of damage",
        "Area = %.2f mm²" % (data[3],),
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        ax.text(0.05, 0.05, textstr, fontsize=14, transform = ax.transAxes,
        va='bottom',ha='left', bbox=props)


        plt.tight_layout()
        plt.savefig("InterrimReport/QuantifiedDamage/"+self.sample+".jpeg",dpi=300)
        plt.show()
        plt.clf()
        
        

class Set(object):
    def __init__(self,set):
        photoNumbers = ["1","2","3","4","5"]
        #photoNumbers = ["4"]
        coupons = []
        damageInfo = []
        for photo in photoNumbers:
            coup = Coupon(set,photo)
            coupons.append(coup)
            coup.getDifference(show=False)
            coup.thresholdImg()
            couponDamage = coup.damageInfo()
            damageInfo.append(couponDamage)
            coup.plots(couponDamage)
            print("Completed Coupon:", set+photo)
        self.df = pd.DataFrame(damageInfo)
        self.df.columns = ['ID', 'Damage Width', 'Damage Height', "Total Area of Damage (mm^2)","Has Preexisting Damage","Vert Arm Length", "Vert Arm Width", "Horizontal Arm Length", "Horizontal Arm Width"]
        #print(df)               


setNames = ["TB","ZA","MA","AS",]
setNames = ["MA"]
setList = []
for set in setNames:
    setList.append(Set(set)) #in a set is 5 coupons, in a coupon is a before and after and difference
    print("Completed Set:", set)
    plt.close('all')
damageDFs = [x.df for x in setList]
damageInfoDF = pd.concat(damageDFs, ignore_index=True)
damageInfoDF.to_csv("DamageData.csv",index=False)
print(damageInfoDF)
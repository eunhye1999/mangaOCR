import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def showimg(filez, title):
    cv2.imshow(title,filez)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("ep1.jpg")
height , width, rgb = img.shape

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgray = cv2.bitwise_not(cv2.adaptiveThreshold(imgray, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 75, 10))

blur = cv2.GaussianBlur(imgray, (5, 5), 0)
thres1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

mask = np.zeros_like(img)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

titles = ['Original Image','Gray Image','Blur Image','BINARY + OTSU']
images = [img, imgray, blur, thres1]

contours, hierarchy = cv2.findContours(thres1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

pruned_contours=[]
for cnt in contours:
    area = cv2.contourArea(cnt)
    # if area > 100 and area < ((height / 3) * (width / 3)):
    if area > 100:
        pruned_contours.append(cnt)

print(len(pruned_contours))
cv2.drawContours(mask, pruned_contours, -1, (255,255,255), 1)
contours2, hierarchy2 = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

showimg(mask,"mask")

# cv2.drawContours(mask, contours2, -1, (255,255,255), 1)
# showimg(mask,"mask")

# print(pruned_contours)
# หาพื้นที่ที่มากพี่สุด คือหา buble

i = 0
for cnt in contours2:
    area = cv2.contourArea(cnt)
    if area > 100 and area < ((height/3) * (width / 3)):
        draw_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)
        
        # cv2.drawContours(img, contours2, i, (0, 255, 0), 2)
        # showimg(img,f"{i}mask")
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        cv2.fillPoly(draw_mask, [approx], (255,0,0))
        image = cv2.bitwise_and(draw_mask, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        showimg(image,"mask")
    
    i+=1
print(i)
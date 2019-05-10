import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
img = cv2.imread("ep1h.jpg")
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(imgray.shape)
blur = cv2.GaussianBlur(imgray, (5, 5), 0)
thres1 = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
    


titles = ['Original Image','Gray Image','Blur Image','BINARY + OTSU']
images = [img, imgray, blur, thres1]

# for i in np.arange(len(titles)):
#   plt.imshow(images[i],'gray')
#   plt.title(titles[i])
#   plt.xticks([]),plt.yticks([])
#   plt.show()

# for i in np.arange(len(titles)):
#     cv2.imshow(titles[i],images[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
 
# contours = cv2.findContours(thres1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
(contours, _) = cv2.findContours(thres1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("tempcon")

print(len(contours))



for (i, c) in enumerate(contours):
    print(i, len(c))
    print(cv2.contourArea(contours[i]))
    cv2.drawContours(img, contours, i, (0, 255, 0), 2)
    picture = img.copy()

    cv2.imshow(f"{i}contours",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

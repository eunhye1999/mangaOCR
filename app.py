import numpy as np
import cv2, os, shutil, time
import matplotlib.pyplot as plt

cwd = os.getcwd()
f_out = os.path.join(cwd, 'output')
def showimg(filez, title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title,filez)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def output_directory(directory):
    list_f = os.listdir(f_out)
    out_dir = os.path.join(f_out, directory)
    if(directory in list_f):
        shutil.rmtree(os.path.join(f_out, directory))
        time.sleep(2)
        os.makedirs(out_dir)
    else:
        os.makedirs(out_dir)
    return out_dir

def get_string(loca_file):
    img = cv2.imread(loca_file)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    height , width, rgb = img.shape
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(imgray, (1, 1), 0)
    showimg(blur,"blur")
    thres1 = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # thres1 = cv2.threshold(imgray, 250, 255, cv2.THRESH_BINARY)[1]
    showimg(thres1,"thres1")

    # contours, hierarchy = cv2.findContours(thres1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thres1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    mask = np.zeros_like(img)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cv2.drawContours(mask, contours, -1, (255,255,255), 1)
    showimg(mask,"mask_black")

    directory = loca_file.split(".")[0]
    path_out = output_directory(directory)
 
    i = 0
    c = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000 and area < ((height/3) * (width / 3)):
            area = cv2.contourArea(cnt)
            draw_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            cv2.fillPoly(draw_mask, [approx], (255,0,0))
            image = cv2.bitwise_and(draw_mask, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            # showimg(image,f"{i} mask")
            cv2.imwrite(f'{path_out}/ep1_{i}.jpg',image) 
            c+=1
        i+=1
    print(i,c)

get_string("mangaep4.jpg")




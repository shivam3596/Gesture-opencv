import socket
import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(1)

host = '127.0.0.1'
port = 8888
 
while True:
    ret, frame= cap.read()
    cv2.rectangle(frame,(300,300),(100,100),(0,255,0),0)
    crop_img = frame[100:300, 100:300]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0 )

    _, thresh1 = cv2.threshold(blur, 100, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)
    #gaus = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,1) 
    #_ ,otsu = cv2.threshold(blur,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    _, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #CHAIN_APPROX_SIMPLE
    max_area = -1
    l = len(contours)
    for i in range(l):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if(area > max_area):
            max_area = area
            ci=i
    cnt = contours[ci]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x,y), (x+w,y+h), (0,0,255), 0)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [cnt],0 ,(0,255,0),0)
    cv2.drawContours(drawing, [hull], 0 ,(0,255,0), 0)
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = -1
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 5)
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]    #[ start point, end point, farthest point, approximate distance to farthest point ]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img,far,1,[0,0,255],-1)
        cv2.line(crop_img,start,end,[0,255,0],2)

    s = socket.socket()
    s.connect((host,port))
    if count_defects == 1:
        message ="open http://www.oldstuff.in"
        print(message)
        s.send(message)
    elif count_defects == 2:
        message ="open http://www.bing.com"
        print(message)
        s.send(message)
    elif count_defects == 3:
        message ="open http://www.youtube.com"
        print(message)
        s.send(message)
    elif count_defects == 4:
        message ="open http://mbasic.facebook.com"
        print(message)
        s.send(message)
    else:
        message ="waiting for instructions..."
        print(message)

    
#Output display of the frames
    cv2.imshow('drwaing', drawing)
    cv2.imshow('vid',frame)    
    cv2.imshow('vid',crop_img)
    #cv2.imshow('gaus',gaus)
#Hold windows unitill q key is pressed  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

s.close()

cap.release()
cv2.destroyAllWindows()

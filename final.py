import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15,15), 0 )
    
    #gaus = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,115,1) 
    _ ,otsu = cv2.threshold(blur,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    _, contours, hierarchy = cv2.findContours(otsu.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 0)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(frame.shape,np.uint8)
    cv2.drawContours(drawing, [cnt],0 ,(0,255,0),0)
    cv2.drawContours(drawing, [hull], 0 ,(0,255,0), 0)
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects =0
    cv2.drawContours(otsu, contours, -3, (0,255,0), 3)
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(frame,far,1,[0,0,255],-1)
        cv2.line(frame,start,end,[0,255,0],2)

    font = cv2.FONT_HERSHEY_SIMPLEX  

    if count_defects == 1:
        cv2.putText(frame, '2 ', (0,130), font, 1, (0,255,0), 3, cv2.LINE_AA)
    elif count_defects == 2:
        cv2.putText(frame, '3', (0,130), font, 1, (0,255,0), 3, cv2.LINE_AA)
    elif count_defects == 3:
        cv2.putText(frame, '4', (0,130), font, 1, (0,255,0), 3, cv2.LINE_AA)
    elif count_defects == 4:
        cv2.putText(frame, '5', (0,130), font, 1, (0,255,0), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'gime me command', (0,130), font, 1, (0,255,0), 3, cv2.LINE_AA)
    

#Output display of the frames
    cv2.imshow('vid',frame)
    #cv2.imshow('gray', gray)
    #cv2.imshow('blur', blur)
    cv2.imshow('otsu thresh',otsu)
    #cv2.imshow('gaus',gaus)
    cv2.imshow('drwaing', drawing)
#Hold windows unitill q key is pressed  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

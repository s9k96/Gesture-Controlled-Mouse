import pymouse
import cv2
import math
import numpy as np

m = pymouse.PyMouse()

hand_cascade = cv2.CascadeClassifier('hand.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 1500)
cap.set(4,700)
grabbed = False
disc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
while cap.isOpened():
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    hand = hand_cascade.detectMultiScale(gray, 1.3, 5)
    grabbed = False
    
    for (x, y, w, h) in hand:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    	roi = frame[y-h:y+h, x-w:x+ 2 *w]
    	roi2 = frame[y:y+h, x:x+w]
        m.move(x, y)

        if roi.size and roi2.size:
            # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
            mean_val = cv2.mean(roi)
            print int(mean_val[0])  
            lower_limit= np.array([int(mean_val[0])-10, 50,50])
            upper_limit= np.array([int(mean_val[0])+10, 255,255])
            mask=cv2.inRange(frame, lower_limit, upper_limit)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            ret2,th2 = cv2.threshold(roi,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            image, contours, hierarchy = cv2.findContours(th2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
            hull = cv2.convexHull(cnt,returnPoints = False)
            defects = cv2.convexityDefects(cnt,hull)
            cv2.drawContours(th2,[cnt],0,(0,0,255),0)

            ##############
            count_defects=0
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
                    cv2.circle(th2,far,1,[0,0,255],-1) 
                #dist = cv2.pointPolygonTest(cnt,far,True)
                cv2.line(th2,start,end,[0,255,0],2)
                #cv2.circle(crop_img,far,5,[0,0,255],-1)
            if count_defects == 1:
                cv2.putText(frame,"Nothing NADA", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            elif count_defects == 2:
                cv2.putText(frame, "something", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            elif count_defects == 3:
                cv2.putText(frame,"This is 4 ", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            elif count_defects == 4:
                cv2.putText(frame,"5 in fifth", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            else:
                cv2.putText(frame,"Can't see anything :(", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)




            # cv2.imshow("frame1", thresh)
            # cv2.imshow("frame2", th2)
            # cv2.imshow("mask", mask)

            cv2.imshow("roi", th2)
            # cv2.imshow("roi2", roi2)

    	grabbed = True

    
    cv2.imshow("frame", frame)
    
    k = cv2.waitKey(10)
    if k == 27 & 0xFF:
        break

cap.release()
cv2.destroyAllWindows()
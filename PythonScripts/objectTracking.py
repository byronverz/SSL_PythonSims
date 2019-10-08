# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 11:44:34 2019

@author: byron
"""

import cv2

tracker = cv2.TrackerCSRT_create()

vid = cv2.VideoCapture(0)

if (vid.isOpened()== False): 
  print("Error opening video stream or file")

bbBoxInit = None
fps = None
#if not vid.isOpen():
#    print("Could not open")
    
#ok, frame = vid.read()

while(vid.isOpened()):
    ok,frame = vid.read()
    (h,w) = frame.shape[:2]
    if bbBoxInit != None:
        (success,box) = tracker.update(frame)
        print(success)
        if success:
            (x,y,w,h) = [int(x) for x in box]
            cv2.rectangle(frame,(x,y),(w+x,h+y),(0,0,255),1)
            
        
    cv2.imshow("Stream",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        bbBoxInit = cv2.selectROI("Stream",frame,fromCenter = False,showCrosshair = True)
        tracker.init(frame,bbBoxInit)
    elif key == ord("x"):
      break

vid.release()
cv2.destroyAllWindows
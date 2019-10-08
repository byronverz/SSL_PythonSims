# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:51:11 2019

@author: byron
"""

import cv2

fps = 10
frames = []

im1 = cv2.imread("frame1.png")
frames.append(im1)

im2 = cv2.imread("frame2.png")
frames.append(im2)

im3 = cv2.imread("frame3.png")
frames.append(im3)

im4 = cv2.imread("frame4.png")
frames.append(im4)

im5 = cv2.imread("frame5.png")
frames.append(im5)

im6 = cv2.imread("frame6.png")
frames.append(im6)

im7 = cv2.imread("frame7.png")
frames.append(im7)

im8 = cv2.imread("frame8.png")
frames.append(im8)

im9 = cv2.imread("frame9.png")
frames.append(im9)

im10 = cv2.imread("frame10.png")
frames.append(im10)

im11 = cv2.imread("frame11.png")
frames.append(im11)

im12 = cv2.imread("frame12.png")
frames.append(im12)

im13 = cv2.imread("frame13.png")
frames.append(im13)

im14 = cv2.imread("frame14.png")
frames.append(im14)

im15 = cv2.imread("frame15.png")
frames.append(im15)

im16 = cv2.imread("frame16.png")
frames.append(im16)

im17 = cv2.imread("frame17.png")
frames.append(im17)

im18 = cv2.imread("frame18.png")
frames.append(im18)

im19 = cv2.imread("frame19.png")
frames.append(im19)

im20 = cv2.imread("frame20.png")
frames.append(im20)

im21 = cv2.imread("frame21.png")
frames.append(im21)

im22 = cv2.imread("frame22.png")
frames.append(im22)

im23 = cv2.imread("frame23.png")
frames.append(im23)

im24 = cv2.imread("frame24.png")
frames.append(im24)

im25 = cv2.imread("frame25.png")
frames.append(im25)

im26 = cv2.imread("frame26.png")
frames.append(im26)

im27 = cv2.imread("frame27.png")
frames.append(im27)

im28 = cv2.imread("frame28.png")
frames.append(im28)

im29 = cv2.imread("frame29.png")
frames.append(im29)

im30 = cv2.imread("frame30.png")
frames.append(im30)

im31 = cv2.imread("frame31.png")
frames.append(im31)

im32 = cv2.imread("frame32.png")
frames.append(im32)

im33 = cv2.imread("frame33.png")
frames.append(im33)

im34 = cv2.imread("frame34.png")
frames.append(im34)

im35 = cv2.imread("frame35.png")
frames.append(im35)

im36 = cv2.imread("frame36.png")
frames.append(im36)

im37 = cv2.imread("frame37.png")
frames.append(im37)

im38 = cv2.imread("frame38.png")
frames.append(im38)

im39 = cv2.imread("frame39.png")
frames.append(im39)

im40 = cv2.imread("frame40.png")
frames.append(im40)

im41 = cv2.imread("frame41.png")
frames.append(im41)

im42 = cv2.imread("frame42.png")
frames.append(im42)

im43 = cv2.imread("frame43.png")
frames.append(im43)

im44 = cv2.imread("frame44.png")
frames.append(im44)

im45 = cv2.imread("frame45.png")
frames.append(im45)

im46 = cv2.imread("frame46.png")
frames.append(im46)

im47 = cv2.imread("frame47.png")
frames.append(im47)

im48 = cv2.imread("frame48.png")
frames.append(im48)

im49 = cv2.imread("frame49.png")
frames.append(im49)

im50 = cv2.imread("frame50.png")
frames.append(im50)

im51 = cv2.imread("frame51.png")
frames.append(im51)

im52 = cv2.imread("frame52.png")
frames.append(im52)

im53 = cv2.imread("frame53.png")
frames.append(im53)

im54 = cv2.imread("frame54.png")
frames.append(im54)

im55 = cv2.imread("frame55.png")
frames.append(im55)

im56 = cv2.imread("frame56.png")
frames.append(im56)

im57 = cv2.imread("frame57.png")
frames.append(im57)

im58 = cv2.imread("frame58.png")
frames.append(im58)

im59 = cv2.imread("frame59.png")
frames.append(im59)

im60 = cv2.imread("frame60.png")
frames.append(im60)

im61 = cv2.imread("frame61.png")
frames.append(im61)

im62 = cv2.imread("frame62.png")
frames.append(im62)

im63 = cv2.imread("frame63.png")
frames.append(im63)

im64 = cv2.imread("frame64.png")
frames.append(im64)

im65 = cv2.imread("frame65.png")
frames.append(im65)

im66 = cv2.imread("frame66.png")
frames.append(im66)

im67 = cv2.imread("frame67.png")
frames.append(im67)

im68 = cv2.imread("frame68.png")
frames.append(im68)

im69 = cv2.imread("frame69.png")
frames.append(im69)

im70 = cv2.imread("frame70.png")
frames.append(im70)

im71 = cv2.imread("frame71.png")
frames.append(im71)

im72 = cv2.imread("frame72.png")
frames.append(im72)

im73 = cv2.imread("frame73.png")
frames.append(im73)
vidOut = cv2.VideoWriter('4MIC_DOA_SIM.avi',cv2.VideoWriter_fourcc('M','J','P','G',),fps,(im1.shape[1],im1.shape[0]))

for i in frames:
    vidOut.write(i)
vidOut.release()
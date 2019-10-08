from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import time

tracker = cv2.Tracker_create("KCF")
initBB = None
vs = VideoStream(src = 0).start()
time.sleep(1.0)
fps = None

while True:
    frame = vs.read()
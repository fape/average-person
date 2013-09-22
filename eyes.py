#!/usr/bin/python
import Image
import sys
import cv2.cv as cv
import string
from random import randrange


imcolor = cv.LoadImage('feher/A2VCE0_IMG_1652.JPG') # input image

# loading the classifiers
haarEyes = cv.Load('/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

# running the classifiers
storage = cv.CreateMemStorage()
detectedEyes = cv.HaarDetectObjects(imcolor, haarEyes, storage)

# draw a purple rectangle where the eye is detected
if detectedEyes:
 for face in detectedEyes:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(155, 55, 200),2)


cv.NamedWindow('Face Detection', cv.CV_WINDOW_AUTOSIZE)
cv.ShowImage('Face Detection', imcolor) 
cv.WaitKey()


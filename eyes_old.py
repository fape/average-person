#!/usr/bin/python
import Image
import sys
import cv2.cv as cv
import string
from random import randrange


imcolor = cv.LoadImage('feher/A2VCE0_IMG_1652.JPG') # input image
# loading the classifiers
haarFace = cv.Load('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
'''haarFace2 = cv.Load('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
haarFace3 = cv.Load('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt_tree.xml')
haarFace4 = cv.Load('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml')
'''
haarEyes = cv.Load('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
haarEyes2 = cv.Load('/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
haarEyes3 = cv.Load('/usr/share/opencv/haarcascades/haarcascade_mcs_eyepair_small.xml')
# running the classifiers
storage = cv.CreateMemStorage()
detectedFace = cv.HaarDetectObjects(imcolor, haarFace, storage)
'''detectedFace2 = cv.HaarDetectObjects(imcolor, haarFace2, storage)
detectedFace3 = cv.HaarDetectObjects(imcolor, haarFace3, storage)
detectedFace4 = cv.HaarDetectObjects(imcolor, haarFace4, storage)
'''
detectedEyes = cv.HaarDetectObjects(imcolor, haarEyes, storage)
detectedEyes2 = cv.HaarDetectObjects(imcolor, haarEyes2, storage)
detectedEyes3 = cv.HaarDetectObjects(imcolor, haarEyes3, storage)
'''
# draw a green rectangle where the face is detected
if detectedFace:
 for face in detectedFace:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(155, 255, 25),2)
# draw a green rectangle where the face is detected
if detectedFace2:
 for face in detectedFace2:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(0, 255, 25),2)
# draw a green rectangle where the face is detected
if detectedFace3:
 for face in detectedFace3:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(255, 255, 25),2)
# draw a green rectangle where the face is detected
if detectedFace4:
 for face in detectedFace4:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(155, 0, 25),2)
'''
# draw a purple rectangle where the eye is detected
if detectedEyes:
 for face in detectedEyes:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(155, 55, 200),2)

# draw a purple rectangle where the eye is detected
if detectedEyes2:
 for face in detectedEyes2:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(155, 255, 200),2)

# draw a purple rectangle where the eye is detected
if detectedEyes3:
 for face in detectedEyes3:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(0, 255, 200),2)

cv.NamedWindow('Face Detection', cv.CV_WINDOW_AUTOSIZE)
cv.ShowImage('Face Detection', imcolor) 
cv.WaitKey()


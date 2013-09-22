#!/usr/bin/python
import Image
import sys
import cv2
import cv2.cv as cv
import string
import os

haarEyes = cv.Load('/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

# running the classifiers
storage = cv.CreateMemStorage()
files = os.listdir("feher")
fnum = len(files)
db = 0
if files:
 for data in files:
  print "image: {}".format(data)
  imcolor = cv.LoadImage("feher/"+data)
  detectedEyes = cv.HaarDetectObjects(imcolor, haarEyes, storage)

  # draw a purple rectangle where the eye is detected
  if detectedEyes and len(detectedEyes)==2:
   for face in detectedEyes:
    angle = 45
    matrix = cv2.getRotationMatrix2D((face[0][0]+face[0][2]/2, face[0][1]+face[0][3]/2), angle, 1.0);
    cv.Circle(imcolor, (face[0][0]+face[0][2]/2, face[0][1]+face[0][3]/2), 10, cv.RGB(255,0,0), 2)
    result = cv2.warpAffine(imcolor, matrix, imcolor.shape, flags=cv2.INTER_LINEAR);
    #cv.Rectangle(imcolor,(face[0][0],face[0][1]),(face[0][0]+face[0][2],face[0][1]+face[0][3]),
    #           cv.RGB(255, 0, 200),2)
   cv.SaveImage("feher_new/"+data, result);
   db = db + 1;

print "{0}/{1} ({2}%)".format(db, fnum, db * 100 / fnum)


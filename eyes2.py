#!/usr/bin/python
import cv2
import os
import time
import numpy as np

start = time.time()  

cascade_fn = "/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
cascade = cv2.CascadeClassifier(cascade_fn)

# running the classifiers

files = os.listdir("feher")
files = sorted(files)
fnum = len(files)
db = 0
acc = np.zeros((600,399,3), np.uint8)
if files:
 for data in files:
#  print "image: {}".format(data)
  img = cv2.imread("feher/"+data)

  detected = cascade.detectMultiScale(img)

  # draw a purple rectangle where the eye is detected
  if len(detected) == 2:
   x1 = detected[0][0] + detected[0][2]/2
   y1 = detected[0][1] + detected[0][3]/2
   x2 = detected[1][0] + detected[1][2]/2
   y2 = detected[1][1] + detected[1][3]/2

   center = ((x1+x2)/2, (y1+y2)/2)
   angle = -2
	   
   cv2.circle(img, (x1, y1), 10, (255,0,0), 2)
   cv2.circle(img, (x2, y2), 10, (0,255,0), 2)
   cv2.circle(img, center, 3, (0,0, 255), -1)
  
   rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
   result = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]),
		borderValue=(107,185,126)); 
   #cv2.accumulate(result, acc);

   cv2.imwrite("feher_new/"+data, result);
   db = db + 1;

end = time.time()
cv2.imwrite("feher_new/00000_acc.jpg", acc);
print "{0}/{1} ({2}%), {3}".format(db, fnum, db * 100 / fnum, end - start)


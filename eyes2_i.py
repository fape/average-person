#!/usr/bin/python
import cv2
import os
import time
import numpy as np
from PIL import Image
import math
import random

start = time.time()  

cascade_fn = "/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
cascade = cv2.CascadeClassifier(cascade_fn)

# running the classifiers
path="ig_resized"
files = os.listdir(path)
#files = sorted(files)
random.shuffle(files)
fnum = len(files)
db = 0
count = 0
w = 450 
h = 700
iw=0
ih=0
acc = Image.new('RGB', (w, h), (0,0,0))
accim = np.array(acc, dtype=np.float32)

if files:
 for data in files:
#  print "image: {}".format(data)
  img = cv2.imread(path+"/"+data)

  detected = cascade.detectMultiScale(img)

  # draw a purple rectangle where the eye is detected
  if len(detected) == 2:
   x1 = detected[0][0] + detected[0][2]/2
   y1 = detected[0][1] + detected[0][3]/2
   x2 = detected[1][0] + detected[1][2]/2
   y2 = detected[1][1] + detected[1][3]/2
   cx = (x1+x2)/2
   cy = (y1+y2)/2	
   iw = img.shape[1]
   ih = img.shape[0]
   angle = math.degrees(math.atan((y1-cy)/float(x1-cx))) 
     	   
  # cv2.circle(img, (x1, y1), 10, (255,0,0), 2)
  # cv2.circle(img, (x2, y2), 10, (0,255,0), 2)
  # cv2.circle(img, (cx, cy), 3, (0,0, 255), -1)
  
   rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
   result = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]),	borderValue=(107,185,126));  
  # cv2.imwrite("feher_new/"+data, result);

   result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
   
   m=Image.fromstring("RGB", (result.shape[1], result.shape[0]), result.tostring())
   	  
   tmp = Image.new('RGB', (w, h), (107,185,126))
   #tmp.paste(m, (0, 0, result.shape[1], result.shape[0]));
   px = w/2 - cx
   py = h/3 - cy
   tmp.paste(m, (px, py, px + result.shape[1], py + result.shape[0]));
   
   #acc = Image.blend(acc, tmp, 0.5)	
   
   accim += np.array(tmp, dtype=np.float32)   
   
   db = db + 1;

  count = count + 1
  print "{}%".format(count* 100 /fnum)

accim /= db * 1.1
acc = Image.fromarray(np.uint8(accim.clip(0,255)))
crop = acc.crop((w/2-iw/2, h/2-ih/2, w/2+iw/2, h/2 + ih/2))
acc.save("out.png", "PNG");
crop.save("out_crop.png", "PNG")
end = time.time()
print "{0}/{1} ({2}%), {3}".format(db, fnum, db * 100 / fnum, end - start)


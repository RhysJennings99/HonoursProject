# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:31:35 2021

@author: baboo
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_1 = cv2.imread('images/2nd.jpg')
img_2 = cv2.imread('images/1st.jpg')
plt.imshow(img_1)
plt.show()

#Convert images to Grey
img1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)

#Use SIFT to identify keypoints within the image
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#Find matching features
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

#See whether the matching features are suitable
good = []
for m in matches:
    if (m[0].distance < 0.5*m[1].distance):
        good.append(m)
matches = np.asarray(good)

#Find Homography Matrix
if (len(matches[:,0]) >= 4):
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
    raise AssertionError('Can’t find enough keypoints.')
    
#Warp the perspectives so the images fit together and stitch together    
dst = cv2.warpPerspective(img_1,H,((img_1.shape[1] + img_2.shape[1]), img_2.shape[0])) #wraped image
dst[0:img_2.shape[0], 0:img_2.shape[1]] = img_2 #stitched image

plt.imshow(dst)
plt.show()
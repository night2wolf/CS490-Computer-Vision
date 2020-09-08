import cv2
import numpy as np
from matplotlib import pyplot as plt
# Gather 100 images from first 15 classes (total of 1500 images)
img = cv2.imread('NWPU-RESISC45/island/island_008.jpg')
#Question 1 a: compute HSV kmeans model of K=64 //TODO

#Question 1 b: Compute a Histogram from images
cv2.imshow('Source Image',img)
plt.hist(img.ravel(),256,[0,256])
plt.show()
#Question 1 c: use Euclidean and KL distance to measure similarity

#Question 1 d: pick 400 random image and compute 1 nearest neighbor
#prediction from hist distance and plot confusion map

#Question 2: Compute Homography on sample images
# Read source image.
im_src = cv2.imread('p1.jpg')
# Four corners of the book in source image
pts_src = np.array([[923, 903], [397, 293], [863, 0],[1466, 431]])
# Read destination image.
im_dst = cv2.imread('p2.jpg')
# Four corners of the book in destination image.
pts_dst = np.array([[231, 319],[1024, 0],[1470, 405],[709, 910]])
# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)  
# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))    
# Display images
cv2.imshow("Source Image", im_src)
cv2.imshow("Destination Image", im_dst)
cv2.imshow("Warped Source Image", im_out)
cv2.waitKey(0)

import cv2
from HW1helpers import centroid_histogram,plot_colors
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance as dist
from scipy import special
import glob

##### Question 2: Compute Homography on sample images //DONE
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
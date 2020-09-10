# HW1.py - TKlinkenberg tkndf@umsystem.edu
# Refer to HW1Helpers.py for references
import cv2
from HW1helpers import centroid_histogram,plot_colors
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance as dist
from scipy import special
import glob

# Gather 100 images from first 15 classes (total of 1500 images)
index = {}
images = {}
#Import image and Convert to RGB
img = cv2.imread('NWPU-RESISC45/island/island_008.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Display original image
plt.imshow(img)
#### Question 1 a: compute HSV kmeans model of K=64 //DONE
img = img.reshape((img.shape[0] * img.shape[1], 3))
clt = KMeans(n_clusters = 64)
clt.fit(img)
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
# show our color chart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
##### Question 1 b: Compute a Histogram from images 
hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
  [0, 256, 0, 256, 0, 256])
hist = cv2.normalize(hist, hist).flatten()
# index[filename] = hist
#### Question 1 c: use Euclidean and KL distance to measure similarity
# //TODO FIX this for dataset.

# initialize the results dictionary
resultsE = {}
# loop over the index
for (k, hist) in index.items():
	# compute the distance between the two histograms
	# using the method and update the results dictionary
	d = dist.euclidean(index["doge.png"], hist)
	resultsE[k] = d
# sort the results
resultsE = sorted([(v, k) for (k, v) in resultsE.items()])
# show the query image
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images["doge.png"])
plt.axis("off")
# initialize the results figure
fig = plt.figure("Results: %s" % ("Euclidean"))
fig.suptitle("Euclidean", fontsize = 20)
# loop over the results
for (i, (v, k)) in enumerate(results):
	# show the result
	ax = fig.add_subplot(1, len(images), i + 1)
	ax.set_title("%s: %.2f" % (k, v))
	plt.imshow(images[k])
	plt.axis("off")
# show the Euclidean method
plt.show()
#### KL Distance (repeat of most of the above code)
# initialize the dictionary dictionary
resultsKL = {}
# loop over the index
for (k, hist) in index.items():
	# compute the distance between the two histograms
	# using the method and update the results dictionary
	d = special.kl_div(index["doge.png"], hist)
	resultsKL[k] = d
# sort the results
resultsKL = sorted([(v, k) for (k, v) in resultsKL.items()])
# show the query image
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images["doge.png"])
plt.axis("off")
# initialize the results figure
fig = plt.figure("Results: %s" % ("KL Distance"))
fig.suptitle("KL Distance", fontsize = 20)
# loop over the results
for (i, (v, k)) in enumerate(results):
	# show the result
	ax = fig.add_subplot(1, len(images), i + 1)
	ax.set_title("%s: %.2f" % (k, v))
	plt.imshow(images[k])
	plt.axis("off")
# show the KL Distance method
plt.show()
##### Question 1 d: pick 400 random image and compute 1 nearest neighbor
#prediction from hist distance and plot confusion map
from sklearn import metrics
actual = [0]
prediction = [0]
metrics.confusion_matrix(actual,prediction)
metrics.classification_report(actual,prediction)

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

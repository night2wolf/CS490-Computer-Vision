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
for image in glob.glob("NWPU-RESISC45/airplane" + "/*.jpg"):
	#Import image and Convert to RGB
	fileName = image[image.rfind("\\")+ 1:]
	image = cv2.imread(image)
	images[fileName] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# Display original image
	# plt.imshow(image)
	#### Question 1 a: compute HSV kmeans model of K=64 //DONE
	img = image.reshape((image.shape[0] * image.shape[1], 3))
	clt = KMeans(n_clusters = 64)
	clt.fit(img)
	##### Question 1 b: Compute a Histogram from images 
	hist = centroid_histogram(clt)
	bar = plot_colors(hist, clt.cluster_centers_)
	# show our color bar
	#plt.figure()
	#plt.axis("off")
	#plt.imshow(bar)
	#plt.show()
	#plt.figure()
	#plt.title("Color Histogram")
	features = []
	hist = cv2.calcHist(image, [0, 1, 2], None, [8, 4, 4],
  	[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()
	index[fileName] = hist
	#plt.plot(hist)
	#plt.show()
	print(fileName)
	
#### Question 1 c: use Euclidean and KL distance to measure similarity
	# //TODO
	# initialize the results dictionary
	resultsE = {}
		# loop over the index
for (k, hist) in index.items():
	# compute the distance between the two histograms
	# using the method and update the results dictionary
	d = dist.euclidean(index["airplane_001.jpg"], hist)
	resultsE[k] = d
# sort the results
resultsE = sorted([(v, k) for (k, v) in resultsE.items()])
# show the query image
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images["airplane_001.jpg"])
plt.axis("off")
# initialize the results figure
fig = plt.figure("Results: %s" % ("Euclidean"))
fig.suptitle("Euclidean", fontsize = 20)
# loop over the results
for (i, (v, k)) in enumerate(resultsE):
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
	d = special.kl_div(index["airplane_001.jpg"], hist)
	resultsKL[k] = d
# sort the results
# resultsKL = sorted([(v, k) for (k, v) in resultsKL.items()])
# show the query image
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(images["airplane_001.jpg"])
plt.axis("off")
# initialize the results figure
fig = plt.figure("Results: %s" % ("KL Distance"))
fig.suptitle("KL Distance", fontsize = 20)
# loop over the results
for (i, (v, k)) in enumerate(resultsKL):
	# show the result
	ax = fig.add_subplot(1, len(images), i + 1)
	ax.set_title("%s: %.2f" % (k, v))
	plt.imshow(images[k])
	plt.axis("off")
# show the KL Distance method
plt.show()


"""
##### Question 1 d: pick 400 random image and compute 1 nearest neighbor
#prediction from hist distance and plot confusion map
from sklearn import metrics
actual = [0]
prediction = [0]
metrics.confusion_matrix(actual,prediction)
metrics.classification_report(actual,prediction)
"""


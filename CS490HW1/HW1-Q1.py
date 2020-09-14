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
	#plt.imshow(image)
	#### Question 1 a: compute HSV kmeans model of K=64 //DONE
	img = image.reshape((image.shape[0] * image.shape[1], 3))
	clt = KMeans(n_clusters = 64)
	clt.fit(img)
	##### Question 1 b: Compute a Histogram from images 
	hist = centroid_histogram(clt)
	#bar = plot_colors(hist, clt.cluster_centers_)
	# show our color bar
	#plt.figure()
	#plt.axis("off")
	#plt.imshow(bar)
	#plt.show()


	features = []
	hist = cv2.calcHist(image, [0, 1, 2], None, [8, 4, 4],
  	[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()
	index[fileName] = hist
	#plt.figure()
	#plt.title("Color Histogram")
	#plt.plot(hist)
	#plt.show()
	print(fileName)
	
#### Question 1 c: use Euclidean distance to measure similarity	
	# initialize the results dictionary
	resultsE = {}
		# loop over the index
for (k, hist) in index.items():
	# compute the distance between the two histograms
	# using the method and update the results dictionary
	# // TODO - This needs to be a random image out of the dataset
	d = dist.euclidean(index["airplane_001.jpg"], hist)
	resultsE[k] = d
# sort the results
resultsE = sorted([(v, k) for (k, v) in resultsE.items()])
# show the query image
fig = plt.figure("Query")
ax = fig.add_subplot(1, 1, 1)
# // TODO : some random image here
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
#plt.show()
## //TODO Print Results, Use to plot confusion matrix.
#for x in range(len(resultsE)): 
    #print(resultsE[x]) 
#for key, value in images.items():
    #print(key, ' : ', value)
indexvalues = []	
#for key, value in index.items():
	#print(key, ' : ', value)



##### Question 1 d: pick 400 random image and compute 1 nearest neighbor
#prediction from hist distance and plot confusion map
# //TODO : Create 15x15 confusion matrix of all classes.
from sklearn import metrics
actual = images.get("airplane_003.jpg")
prediction = images.get("airplane_023.jpg")
metrics.confusion_matrix(actual,prediction)
metrics.classification_report(actual,prediction)

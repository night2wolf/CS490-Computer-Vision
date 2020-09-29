"""
NOTE for Grader to get vlfeat:
Needs 3.6 python environment and numpy and cython.
then run:
conda install -c menpo cyvlfeat
References:
https://github.com/menpo/cyvlfeat
https://github.com/Lithogenous/VLAD-SIFT-python/blob/master/vlad_raw.py
https://gist.github.com/danoneata/9927923 
https://github.com/menpo/cyvlfeat/blob/master/cyvlfeat/sift/dsift.py
https://github.com/menpo/cyvlfeat/blob/master/cyvlfeat/sift/sift.py 
https://github.com/Vectorized/Python-KD-Tree/blob/master/kdtree.py
"""
from cyvlfeat import vlad,fisher,kmeans,sift
import cv2
import sklearn
import HW2helpers
import scipy
import numpy as np
#Q1
"""
Compute Image Features, create a matlab/python function that compute n x d features by calling
vl_feat DSIFT and SIFT functions (notice that vl_feat also has Python version),
implementing the following function:
"""
#  im - input images, let us make them all grayscale only, so it is a  h x w matrix
#  opt.type = { ‘sift’, ‘dsft’} for sift and densesift
#  feature - n x d matrix containing n features of d dimension
# NOTE - im must be A single channel, greyscale, `float32` numpy array (ndarray)
#  representing the image to calculate descriptors for for sift and dsift
def getImageFeatures(im, opt):
    
    im =cv2.imread(im)    
    img_gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    if opt == "dsift" or opt == "sift":
        if opt == "dsift":
            features = sift.dsift(img_gray)
        if opt == "sift":
            features = sift.sift(img_gray)
    else:
        print("only supports dsift and sift -- ERROR")
    return features


# Q2
"""
Compute VLAD and Fisher Vector models of image features,
for this purpose you need to first compute a Kmeans model for DenseSIFT and SIFT.
Use the CDVS data set given as both training and testing for convenience 
(not the right way in research though, should use a different data set, say FLICKR MIR, or ImageNet),
implementing the following functions: 
"""
# f - n x d matrix containing training features from say 100 images. 
# k - VLAD kmeans model number of cluster 
# kd - desired dimension of the feature 
# vlad_km - VLAD kmeans model
# A - PCA projection for dimension reduction
def getVladModel(f, kd, k):    
    centers = kmeans.kmeans(f,k)
    print(centers)
    kdtree = scipy.spatial.KDTree(centers)    
    #knn = scipy.spatial.KDTree.query(kdtree,1)
    vlad_km = vlad.vlad(f,centers,kdtree)        
    return vlad_km
"""  -- Do the same for Fisher Vector model """
# f - n x d matrix containing n features from say 100 images. 
# k - number of GMM components 
# kd - desired lower dimension of the feature
# fv_gmm - FisherVector GMM model:
#         fv_gmm.m - mean, fv_gmm.cov - variance, fv_gmm.p - prior
# A - PCA for dimension reduction
def getFisherVectorModel(f, kd, k):
    fisher_km = fisher.fisher()
    return fisher_km


#Q3
"""
Compute VLAD and Fisher Vector Aggregation of Images,
from the given VLAD and FV models, implementing the following functions.
Notice that the feature nxd f need to be projected to the desired lower dimension via,
f0=f*A(:,1:kd), to match the VLAD model dimension before calling this function. 
"""
# f - n x d matrix containing a feature from an image by calling f=getImageFeature(im,.. 
# vlad_km - VLAD kmeans model
def getVladAggregation(vlad_km, f):
    return
""" --Do the same for Fisher Vector mdoel """
# f - n x d matrix containing a feature from an image by calling f=getImageFeature(im,.. 
# fv_gmm - GMM Model from features, has m, cov, and p. 
def getFisherVectorAggregation(fv_gmm, f):
    return


#Q4
"""
Now benchmarking the TPR-FPR performance of various feature and aggregation scheme performance against
the mini CDVS data set. For the SIFT and DenseSIFT features, let us have kd=[24, 48], nc=[32, 64, 96].
So for each image, we will have 2 x 6 x2  = 24 different feature + aggregation representations.
For the total of N images in the mini CDVS dataset, we have M=N*(N-1)/2 total image pairs,
and the matching pairs ground truth are given, we only care about the first 100 matching pairs and
first 100 non-matching pairs in the fid.mat, which has two variables mp and nmp.
Each has a row of two image filenames to their associated images, e.g, mp(1,:): mp_2.jpg and mp1_2.jpg
are two matching pairs: 
And nmp(1,:) contains file names for the following non-matching pairs 
Let us use the Euclidean distance to on those features to compute the TPR-FPR ROCs
and find out which one have the best performance. Last year’s plots are attached below for example,
you only need to plot the last row for 10 feature-aggregation combinations.
"""    
kd = [24,48]
nc = [32,64,96]
im = "MPEG.CDVS\cdvs_thumbnails\cdvs-11.jpg"
frames = getImageFeatures(im,opt="sift")
print(frames)
#vlad_km = getVladModel(frames,kd,10)
#print(vlad_km)
frames,descriptor = getImageFeatures(im,opt="dsift")
print(frames)
print(descriptor)
vlad_km = getVladModel(frames,kd,1)
print(vlad_km)
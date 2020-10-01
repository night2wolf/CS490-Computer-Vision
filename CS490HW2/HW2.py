"""
NOTE for Grader to get vlfeat:
Needs 3.6 python environment and numpy and cython.
then run:
conda install -c menpo cyvlfeat
Additional NOTE:
Feature aggregation is library found here:
https://github.com/paschalidoud/feature-aggregation 
run setup.py install as administrator

References:
https://github.com/menpo/cyvlfeat
https://github.com/Lithogenous/VLAD-SIFT-python/blob/master/vlad_raw.py
https://gist.github.com/danoneata/9927923 
https://github.com/Vectorized/Python-KD-Tree/blob/master/kdtree.py
"""
from cyvlfeat import vlad,fisher,kmeans,sift,gmm
import cv2
from sklearn import metrics,svm,pipeline
import scipy
import numpy as np
import matplotlib.pyplot as plt
from feature_aggregation import FisherVectors,Vlad,BagOfWords
import glob
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
    print(im)
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
    fv_gmm = gmm.gmm(f,n_clusters=k)
    print(fv_gmm)        
    fv_gmmM= fv_gmm[0]
    fv_gmmC = fv_gmm[1]
    fv_gmmP = fv_gmm[2]
    print(fv_gmmC)
    print(fv_gmmP)      
    #fisher_km =""
    fisher_km = fisher.fisher(f,fv_gmmM,fv_gmmC,fv_gmmP)
    return fisher_km


#Q3 & Q4  
im = "MPEG.CDVS\cdvs_thumbnails\cdvs-11.jpg"
imgs = "MPEG.CDVS\cdvs_thumbnails"
# Gather all features in an array
dsftfeatures = [
    getImageFeatures(imge,opt="dsift")
    for imge in glob.glob("MPEG.CDVS\cdvs_thumbnails" + "/*.jpg")    
]

sftfeatures = [
    getImageFeatures(imge,opt="sift")
    for imge in glob.glob("MPEG.CDVS\cdvs_thumbnails" + "/*.jpg")    
]
# Fit Vlad and Fisher Vector Models based on Sift and Dsift Features
sfttrain = np.arange(len(sftfeatures))
dsfttrain = np.arange(len(dsftfeatures))
#Shuffle the data set for training and test batches
np.random.shuffle(sfttrain)
np.random.shuffle(dsfttrain)
sfttest = sfttrain[100:]
dsfttest = dsfttrain[100:]
sfttrain = sfttrain[:100]
dsfttrain = dsfttrain[:100]
# Create Fisher Vector and Vlad Models then fit them on the data (see Feature_Aggregation for functions)
sftfv = FisherVectors(100)
sftfv.fit(sftfeatures)
sftfv.transform(sftfeatures)
sftvlad = Vlad(100)
sftvlad.fit(sftfeatures)
sftvlad.transform(sftfeatures)
dsftfv = FisherVectors(100)
dsftfv.fit(dsftfeatures)
dsftfv.transform(dsftfeatures)
dsftvlad = Vlad(100)
dsftvlad.fit(sftfeatures)
dsftvlad.transform(sftfeatures)

"""
vlad_km = getVladModel(features,kd,24)
print(vlad_km)
fisher_km = getFisherVectorModel(features,kd,24)
print(fisher_km)
"""
#final measurement roc curve use this:
# https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html#sklearn.metrics.plot_roc_curve

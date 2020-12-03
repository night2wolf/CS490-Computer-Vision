import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from cyvlfeat import vlad,fisher,kmeans,sift,gmm
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, homogeneity_score, completeness_score,v_measure_score
from PIL import Image
from torchvision import transforms
from feature_aggregation import FisherVectors,Vlad,BagOfWords

#creating an object for VGG16 model(pre-trained)
model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
model.eval()

X_array = []
Y_Array = np.array([])
classes_array = np.array([])

# Single image for testing, loop over entire dataset here
for img in glob.glob("NWPU-RESISC45" + "/*/*.jpg"):    
# Comment this for loop when ready for entire dataset and fix indenting
#img = "NWPU-RESISC45\\airplane\\airplane_635.jpg"
# ----------------
    label = os.path.dirname(img)
    label = os.path.basename(label)
    classes_array = np.append(classes_array,label)
    
    fileName = img[img.rfind("\\")+1:]
    img = Image.open(img)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    output = model(input_batch)    
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # get probabilities, run softmax.
    probs = torch.nn.functional.softmax(output[0], dim=0)
    probs = probs.detach().numpy()
    prob_array = np.array([])
    prob_array = np.append(prob_array,probs)
    X_array.append(prob_array)
    print(fileName)

#- ------- Define Fisher Vector Model--------------
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
# Split into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X_array,classes_array)
fv = FisherVectors(100)
fv.fit(X_array)
fv.transform(X_array)
fisher_km = getFisherVectorModel(features,kd,24)
print(fisher_km)

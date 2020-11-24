# References:
# https://kgptalkie.com/image-classification-using-pre-trained-vgg-16-model/
# https://pytorch.org/hub/pytorch_vision_vgg/
#https://medium.com/analytics-vidhya/cnn-transfer-learning-with-vgg16-using-keras-b0226c0805bd
#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
import os
import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, homogeneity_score, completeness_score,v_measure_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import AgglomerativeClustering
from matplotlib.ticker import NullFormatter
from PIL import Image
from torchvision import transforms
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
#creating an object for VGG16 model(pre-trained)
model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
model.eval()

X_array = []
Y_Array = np.array([])
classes_array = np.array([])

# Single image for testing, loop over entire dataset here
for img in glob.glob("RESISC45\\downloads\\manual\\NWPU-RESISC45" + "/*/*.jpg"):    
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



# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_array,classes_array)
# Run PCA on data for classification report on the classes
pca = PCA().fit(X_train)
plt.figure(figsize=(18, 7))
plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3)
plt.show()
print(np.where(pca.explained_variance_ratio_.cumsum() > 0.95))
pca = PCA(n_components=35).fit(X_train)
X_train_pca = pca.transform(X_train)
classifier = SVC().fit(X_train_pca, y_train)
X_test_pca = pca.transform(X_test)
predictions = classifier.predict(X_test_pca)
print(classification_report(y_test, predictions))
# Run LDA Against the Data

sc = StandardScaler()
X_train_lda = sc.fit_transform(X_train)
X_test_lda = sc.transform(X_test)
lda = LDA(n_components=13)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
classifier = SVC().fit(X_train_lda, y_train)
predictions = classifier.predict(X_test_lda)
print(classification_report(y_test, predictions))
# Now Laplacian Embedding
# According to documentation : Note : Laplacian Eigenmaps is the actual algorithm implemented here.
lpp = SpectralEmbedding(n_components=133)
model = lpp.fit_transform(X_train)
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(133, projection='3d')
ax.scatter(model[:, 0], model[:, 1], model[:, 2],cmap=plt.cm.Spectral)
ax.view_init(4, -72)
ax.set_title("Spectral Embedding")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')
plt.show()


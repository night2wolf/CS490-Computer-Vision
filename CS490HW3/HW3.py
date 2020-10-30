import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import loadmat
from scipy import sparse
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
import HW3helper
# Sources:
#https://towardsdatascience.com/how-to-load-matlab-mat-files-in-python-1f200e1287b5
#https://towardsdatascience.com/eigenfaces-face-classification-in-python-7b8d2af3d3ea
#https://towardsdatascience.com/graph-laplacian-and-its-application-in-machine-learning-7d9aab021d16

#Load matlab data file so we know what we are working with
data = loadmat('HW-3_ Subspace Models/faces-ids-n6680-m417-20x20.mat')
print(data.keys())
# output: dict_keys(['__header__', '__version__', '__globals__', 'faces', 'ids'])
print(type(data['ids']),data['ids'].shape)
#output: <class 'numpy.ndarray'> (6680, 1)
print(type(data['faces']),data['faces'].shape)
#output: <class 'numpy.ndarray'> (6680, 400)
print(type(data['ids'][0][0]),data['ids'][0][0].shape)
#output:  <class 'numpy.uint16'> ()
print(type(data['faces'][0][0]),data['faces'][0][0].shape)
#output: <class 'numpy.float64'> ()
# Create Pandas dataframes out of the matlab data
feature_c = [f'col_{num}' for num in range(400)]
df_features = pd.DataFrame(data=data['faces'],columns=feature_c)
print(df_features)
columns = ['face_ID']
df_ids = pd.DataFrame(data=data['ids'],columns=columns)
print(df_ids)
# Concatenate the IDs with the face features in a single dataframe
df = pd.concat([df_ids,df_features],axis=1,sort=False)
print(df)
#now that we have a dataframe we can display some faces.
print(df['face_ID'].nunique())
def plot_faces(pixels):
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(20, 20), cmap='gray')
    plt.show()
X = df.drop('face_ID',axis=1)
Y = df['face_ID']
# plot_faces(X)
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y)
# now do PCA Eigenfaces
pca = PCA().fit(X_train)
plt.figure(figsize=(18, 7))
plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3)
plt.show()
print(np.where(pca.explained_variance_ratio_.cumsum() > 0.95))
pca = PCA(n_components=133).fit(X_train)
X_train_pca = pca.transform(X_train)
classifier = SVC().fit(X_train_pca, y_train)
X_test_pca = pca.transform(X_test)
predictions = classifier.predict(X_test_pca)
print(classification_report(y_test, predictions))
# Now do LDA Fisher face 
sc = StandardScaler()
X_train_lda = sc.fit_transform(X_train)
X_test_lda = sc.transform(X_test)
lda = LDA(n_components=133)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
classifier = SVC().fit(X_train_lda, y_train)
predictions = classifier.predict(X_test_lda)
print(classification_report(y_test, predictions))
# Now Laplacian Face
# According to documentation : Note : Laplacian Eigenmaps is the actual algorithm implemented here.
lpp = SpectralEmbedding(n_components=133)
model = lpp.fit_transform(X_train)
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(133, projection='3d')
ax.scatter(model[:, 0], model[:, 1], model[:, 2],cmap=plt.cm.Spectral)
ax.view_init(4, -72)
ax.set_title("Spectral Embedding")
from matplotlib.ticker import NullFormatter
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')
plt.show()
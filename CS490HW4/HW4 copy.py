# References:
# https://kgptalkie.com/image-classification-using-pre-trained-vgg-16-model/
# https://pytorch.org/hub/pytorch_vision_vgg/
#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
import os
from glob import glob
import cv2
import numpy as np
import matplotlib as plt
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Folder names give us our label names
classes_array = os.listdir('RESISC45\\downloads\\manual\\NWPU-RESISC45')
valid_path = 'RESISC45\\downloads\\manual\\NWPU-RESISC45'
train_path = 'RESISC45\\downloads\\manual\\NWPU-RESISC45'
epochs = 5
IMAGE_SIZE = [256,256]
batch_size=32
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')
extra_array = os.listdir('Extra Data')
classes_array = classes_array + extra_array
print(classes_array)
folders = glob(train_path + '/*')
#creating an object for VGG16 model(pre-trained)
model = VGG16(include_top=False,weights='imagenet',input_shape=(256,256,3))
"""
ds = tfds.load('resisc45', split='train', shuffle_files=True,data_dir='RESISC45')
ds = ds.shuffle(1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
for element in ds.take(1):
    image, label = element["image"],element["label"]
    print(image)
    print(label)
"""
for layer in model.layers:
    layer.trainable = False
x = Flatten()(model.output)
prediction=Dense(len(folders),activation='softmax')(x)
model=Model(inputs=model.input,outputs=prediction)
model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
gen=ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.2,horizontal_flip=True,vertical_flip=True,preprocessing_function=preprocess_input)
test_gen=gen.flow_from_directory(valid_path,target_size=IMAGE_SIZE)
print(test_gen.classes[49])
labels = [None] * len(test_gen.class_indices)
print(labels)
for k, v in test_gen.class_indices.items():
  labels[v] = k
print(labels[0])
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
# fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)
def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm

cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)
# //TODO : put above into format for PCA()

# Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(image,label)
# Run PCA on data for classification report on the classes
"""
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
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')
plt.show()
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import sklearn.metrics as metrics
from keras.utils import to_categorical
from keras.models import load_model


# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[69]:


# Splitting the original "Training" set of the Fruits360 in order to have some images for the Test set.
# Therefore, the original "Test" set of the Fruits360 will become our Validation set.
# import split_folders
# split_folders.ratio('D:/Fruits360/Training', output="output", seed=1337, ratio=(.85, .15))


# In[71]:


# This was the initial code to obtain the new dataset in which we have not considered "Multiple Fruits".
# I saved these three sets into a folder called "Fruits-360" and using this new data for modelling.

# datagen = tf.keras.preprocessing.image.ImageDataGenerator()
# train = datagen.flow_from_directory('C:/Users/vidya/Documents/Code/output/train', class_mode='categorical', batch_size=32)
# test = datagen.flow_from_directory('C:/Users/vidya/Documents/Code/output/val', class_mode='categorical', batch_size=32)
# val = datagen.flow_from_directory('D:/Fruits360/Test', class_mode='categorical', batch_size=32)


# In[38]:


# Let's understand the number of images we have per class in the train set and the test set.
base_path = "D:/Fruits-360/"
for fruit in os.listdir(base_path + "train"):
    print(str(len(os.listdir(base_path + "train/" + fruit))) + " " + fruit)


# In[5]:


base_path = "D:/Fruits-360/"
for fruit in os.listdir(base_path + "val"):
    print(str(len(os.listdir(base_path + "val/" + fruit))) + " " + fruit)


# In[11]:


# This determines the shape of the image
image = cv2.imread('D:/Fruits-360/train/Apple Braeburn/106_100.jpg')
print(image.shape)


# In[12]:


X_train = [] 
y_train = [] 
k = 0

base_path = "D:/Fruits-360/"
for fruit in os.listdir(base_path + "train"):
    for img in glob.glob(os.path.join(base_path + "train/" + fruit, "*.jpg")):
        img = cv2.imread(img)
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_train.append(img)
        y_train.append(k)
    k = k + 1   #storing class indices as the labels
        
X_train=np.array(X_train)
y_train=np.array(y_train)
y_train=to_categorical(y_train, num_classes=120)

X_train.shape


# In[13]:


y_train.shape


# In[14]:


y_train


# In[16]:


X_val = [] 
y_val = []
k = 0

base_path = "D:/Fruits-360/"
for fruit in os.listdir(base_path + "val"):
    for img in glob.glob(os.path.join(base_path + "val/" + fruit, "*.jpg")):
        img = cv2.imread(img)
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_val.append(img)
        y_val.append(k)
    k = k + 1    #storing class indices as the labels
    
X_val=np.array(X_val)
y_val=np.array(y_val)
y_val = to_categorical(y_val, 120)

X_val.shape


# In[17]:


y_val.shape


# In[18]:


y_val


# In[19]:


X_test = [] 
y_test = []
k = 0

base_path = "D:/Fruits-360/"
for fruit in os.listdir(base_path + "test"):
    for img in glob.glob(os.path.join(base_path + "test/" + fruit, "*.jpg")):
        img = cv2.imread(img)
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_test.append(img)
        y_test.append(k)
    k = k + 1   #storing class indices as the labels
    
X_test=np.array(X_test)
y_test=np.array(y_test)

X_test.shape


# In[20]:


y_test.shape


# In[21]:


y_test


# In[22]:


# Our first model is the VGG19 model
vgg19 = keras.applications.vgg19
conv_model = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(100,100,3))
for layer in conv_model.layers: 
    layer.trainable = False
x = keras.layers.Flatten()(conv_model.output)
x = keras.layers.Dense(100, activation='relu')(x)
predictions = keras.layers.Dense(120, activation='softmax')(x)
full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)
full_model.summary()


# In[17]:


# The second model is the Resnet50 model
conv_model = keras.applications.resnet50.ResNet50(weights= "imagenet", include_top=False, input_shape= (256,256,3))
for layer in conv_model.layers: 
    layer.trainable = False
x = conv_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
predictions = keras.layers.Dense(120, activation= 'softmax')(x)
full_model2 = keras.models.Model(inputs = conv_model.input, outputs = predictions)
full_model2.summary()


# In[24]:


# We will first compile the VGG19 model because VGG19 provides better prediction compared to Resnet50
from PIL.Image import core as _imaging
full_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adamax(lr=0.001), metrics=['acc'])

callback = [
    keras.callbacks.ModelCheckpoint(
        filepath='newmodel1',
        save_best_only=True, monitor='val_loss', verbose=1)]

history = full_model.fit(X_train, y_train, epochs=5, verbose=1, callbacks=callback, validation_data=(X_val,y_val))


# In[25]:


def plot_history(history, yrange):
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Get number of epochs
    epochs = range(len(acc))
    
    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)    
    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')    
    plt.show()


# In[26]:


plot_history(history, yrange=(0.2,1.05))


# In[68]:


# Loading the saved model
full_model = load_model('newmodel1')


# In[27]:


Y_pred = full_model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)


# In[28]:


plt.imshow(X_test[25])


# In[29]:


y_pred[25]


# In[32]:


# Converting Confusion matrix to a DataFrame
cm = metrics.confusion_matrix(y_test, y_pred)
df = pd.DataFrame(cm)
print('Confusion Matrix')
df


# In[6]:


#df.to_csv("Conf_Mat.csv")


# In[34]:


print('Classification Report')
print(metrics.classification_report(y_test, y_pred))


# In[36]:


import matplotlib.pyplot as plt
cm = metrics.confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()


# In[14]:


from numpy import argmax
label = (test.class_indices)
def which_fruit(pred, label):
    inverted = argmax(pred)
    for key in label:
        if label[key] == inverted:
            return key


# In[21]:


image = load_img('D:/Fruits-360/test/Apple Crimson Snow/17_100.jpg', target_size=(256,256))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)


# In[22]:


pred = full_model.predict(image)
which_fruit(pred, label)


# In[ ]:





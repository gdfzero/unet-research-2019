#1048576/512/512 = 4
# -*- coding: utf-8 -*-
"""UNET5-colab.ipynb
"""
#CUDA_VISIBLE_DEVICES=0
import os

import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import Concatenate
from keras.models import Sequential
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense
from keras.models import Model
import matplotlib.image as mpimg
from glob import glob
from keras.preprocessing import image
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from PIL import Image
from keras.models import load_model



#import sys
#sys.exit()

#print("len-----",len(os.listdir('./CT_data/ndct/train/')))


ndct = sorted(glob('/data/CT_data/images/ndct/train/*.flt'))
ldct = sorted(glob('/data/CT_data/sparseview/sparseview_60/train/*.flt'))

ndct_test = sorted(glob('/data/CT_data/images/ndct/test/*.flt'))
ldct_test = sorted(glob('/data/CT_data/sparseview/sparseview_60/test/*.flt'))
#print("ldct_test", len(ldct_test))

# load model
#model = load_model('model.h5')

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    maxval = np.amax(im1)
    psnr = 10 * np.log10(maxval ** 2 / mse)
    return psnr

def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    #mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    mse = tf.compat.v1.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))
import struct

ndct_imgs_train = []
for i in range(0, len(ndct)):                                                                                                                                      
#for i in range(0, 3600):
    f = open(ndct[i],'rb')
    a = np.fromfile(f, np.float32)
    img = a.astype(float)
    img = np.reshape(img,(512,512,1))
    #print("img.shape",img.shape)
    #print("img.dtype", img.dtype)
    ndct_imgs_train.append(img)
    f.close()
print("len(ndct_imgs_train)....",len(ndct_imgs_train))
print("-----")
                                                                                                                                                         
ldct_imgs_train = []
for i in range(0, len(ldct)):
#for i in range(0, 3600):
    f = open(ldct[i],'rb')
    a = np.fromfile(f, np.float32)
    img = a.astype(float)
    img = np.reshape(img,(512,512,1))
    #print("img.shape",img.shape)
    #print("img.dtype", img.dtype)
    ldct_imgs_train.append(img)
    f.close()
print("len(ldct_imgs_train)....",len(ldct_imgs_train))

ndct_imgs_test = []
for i in range(0, len(ndct_test)):
#for i in range(0, len(ndct_test)):
    f = open(ndct_test[i],'rb')
    a = np.fromfile(f, np.float32)
    img = a.astype(float)
    img = np.reshape(img,(512,512,1))
    #print("img.shape",img.shape)
    #print("img.dtype", img.dtype)
    ndct_imgs_test.append(img)
    f.close()
print("len(ndct_imgs_test)......",len(ndct_imgs_test))


# load the image
ldct_imgs_test = []
for i in range(0, len(ldct_test)):
    f = open(ldct_test[i],'rb')
    a = np.fromfile(f, np.float32)
    img = a.astype(float)
    img = np.reshape(img,(512,512,1))

    #print("img.shape",img.shape)
    #print("img.dtype", img.dtype)
    ldct_imgs_test.append(img)
    f.close()
print("len(ldct_imgs_test).......: ",len(ldct_imgs_test))


k1 = np.asarray(ldct_imgs_train)
k2 = np.asarray(ndct_imgs_train)
k1 = k1[:,:,:,0]
k2 = k2[:,:,:,0]
k1 = k1.reshape(3600,512,512,1)
k2 = k2.reshape(3600,512,512,1)
print(k1.shape)
print(k2.shape)
k3 = np.asarray(ldct_imgs_test)
k4 = np.asarray(ndct_imgs_test)
print("k3.shape",k3.shape)
k3 = k3[:,:,:,0]
k4 = k4[:,:,:,0]
k3 = k3.reshape(len(ldct_imgs_test),512,512,1)
k4 = k4.reshape(len(ldct_imgs_test),512,512,1)
print(k3.shape)
print(k4.shape)
print("---------------------------------------------------------")

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
history = LossHistory()

# https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data
from keras.layers.convolutional import Conv2DTranspose
from keras.layers import concatenate

inputs = Input((None, None,1))

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)


c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
p5 = MaxPooling2D(pool_size=(2, 2)) (c5)

c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (c55)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)
u6 = concatenate([u6, c5])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)


u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

output_img = Conv2D(1, (1, 1)) (c9)
subtracted = keras.layers.Subtract()([inputs, output_img])


model = Model(inputs=[inputs], outputs=[subtracted])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
model.compile(optimizer='adam', loss='mse', metrics=[tf_psnr])



#model = load_model('model.h5')

model.fit(k1, k2, validation_split=.1, batch_size=30, epochs=50, callbacks=[history])

#print(history.losses)

reconstructed = model.predict(k3)
psnr = cal_psnr(k4, reconstructed)
print("psnr 50 epochs.....",psnr)


a = reconstructed[0].reshape(512, 512)
scalef = np.amax(a)
a = np.clip(255 * a/scalef, 0, 255).astype('uint8')
#result = Image.fromarray((a * 255).astype(np.uint8))                                                                                                
result = Image.fromarray((a).astype(np.uint8))
result.save('rec_0_50.png')



# save image 12 because it is the hardest to reconstruct
a = reconstructed[12].reshape(512, 512)
scalef = np.amax(a)
a = np.clip(255 * a/scalef, 0, 255).astype('uint8')
#result = Image.fromarray((a * 255).astype(np.uint8))                                                                             
result = Image.fromarray((a).astype(np.uint8))
result.save('rec_12_50.png')


model.save("model.h5")


# Score trained model.
scores = model.evaluate(k3, k4, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

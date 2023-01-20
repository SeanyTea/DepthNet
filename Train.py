from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.models import Model
import numpy as np
from scipy.ndimage import rotate
import tensorflow.keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import scipy.io
import math
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D,Add,SeparableConv2D, MaxPooling2D,concatenate,ZeroPadding2D,Cropping2D,Dropout,Lambda,Reshape,Input,Concatenate, concatenate,Conv3D,BatchNormalization,Activation,UpSampling2D,Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model,Model
from skimage import data, img_as_float
from skimage import exposure
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.pylab as plt
import numpy as np
import random
import scipy
import cv2 as cv
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage
from tensorflow.keras.models import Sequential, load_model,Model
from scipy import *
from utils import *
import imageio
import os
from keras.layers import LeakyReLU
import torch
import keras.backend as K

# * Define a custom loss function
def custom_loss(y_true,y_pred):
	lmbda_1 = 1
	lmbda_2 = 1.3
	#find positive points
	#Copy arrays
	print(y_true.shape)
	y_true = y_true.numpy()
	y_pred = y_pred.numpy()
	y_true_pos = y_true.clone()
	y_true_neg = y_true.clone()
	y_pred_pos = y_pred.clone()
	y_pred_neg = y_pred.clone()

	# Get arrays
	y_true_pos[y_true_pos <= 0] = 0
	y_true_neg[y_true_neg > 0] = 0
	y_pred_pos[y_pred_pos <= 0] = 0
	y_pred_neg[y_pred_neg > 0] = 0

	loss = lmbda_1 * np.mean(np.square(y_true_pos - y_pred_pos), axis=-1) \
		+ lmbda_2 * np.mean(np.square(y_true_neg - y_pred_neg), axis=-1)
	return loss

def custom_mse(y_true,y_pred):
	lmbda_1 = 1
	lmbda_2 = 1.3
	y_true_pos = tf.cast(y_true > 0, y_true.dtype) * y_true
	y_true_neg = tf.cast(y_true <=0,y_true.dtype) * y_true
	y_pred_pos = tf.cast(y_pred > 0, y_pred.dtype) * y_pred
	y_pred_neg = tf.cast(y_pred <=0,y_pred.dtype) * y_pred

	loss = lmbda_1*K.mean(K.square(y_true_pos-y_pred_pos)) + lmbda_2*K.mean(K.square(y_true_neg-y_pred_neg))
	return loss
'''
def custom_mse(y_true, y_pred):
    # calculating squared difference between target and predicted values 
    y_true_pos = tf.cast(y_true > 0, y_true.dtype) * y_true
    print(tf.reduce_max(y_true_pos))
    # multiplying the values with weights along batch dimension
    loss = [0.3, 0.7]          # (batch_size, 2)
                
    # summing both loss values along batch dimension 
    loss = K.sum(loss, axis=1)        # (batch_size,)
    print(loss)
    return loss
'''

#VERSION=0# FULL DPDNET
VERSION=1# FAST VERSION
batch_size = 15
epochs=30
aspect_ratio=480/640 #ASPECT RATIO OF KINECT V2
img_x=320#Kinect V2 half width
img_y=round(aspect_ratio*img_x)
path='/Users/seanthammakhoune/Documents/DPDNet/GOTPD_DATABASE/'
lengthdataset=len(os.listdir(path+'TRAIN_DATA/INPUT'))

if(VERSION==0):
	divider = 1
	canales = 1
	image_input = Input(shape=(int(img_y / divider), int(img_x / divider), 1))
	x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(image_input)
	x = BatchNormalization(axis=3, name='bn_conv1')(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)#Activation('relu')(x) # ! change to Leaky RELU	
	x = MaxPooling2D((3, 3))(x) # * This is fine

	x = encoding_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))

	x = encoding_conv_block(x, 3, [128, 128, 512], stage=3, block='a')

	x = encoding_conv_block(x, 3, [256, 256, 1024], stage=4, block='a')

	x = decoding_conv_block(x, 3, [1024, 1024, 256], stage=5, block='a', strides=(1, 1))

	x = decoding_conv_block(x, 3, [512, 512, 128], stage=6, block='a')

	x = decoding_conv_block(x, 3, [256, 256, 64], stage=7, block='a')
	x = Cropping2D(cropping=((0, 0), (1, 1)), data_format=None)(x)

	x = UpSampling2D(size=(3, 3))(x)
	x = Conv2DTranspose(64, (7, 7), strides=(2, 2), padding='same', name='co')(x)
	x = Cropping2D(cropping=((0, 0), (2, 2)), data_format=None)(x)
	x = BatchNormalization(axis=3, name='bn_c1')(x)
	x = Activation('relu')(x)
	x = Conv2DTranspose(1, (3, 3), padding='same', name='c8o')(x)
	x = Activation('sigmoid')(x)
	x2=tensorflow.keras.backend.concatenate([x,image_input],axis=-1)
	refinement1 = refunit(divider, canales + 1,img_y,img_x)
	x2=refinement1(x2)
	x2 = ZeroPadding2D(padding=((2, 2), (0,0)), data_format=None)(x2)

	x2 = x + x2
	model = Model(inputs=image_input, outputs=[x,x2])
	model.summary()
	check = tensorflow.keras.callbacks.ModelCheckpoint('DPDnet.h5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='auto', period=1)
	model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),loss=['mse','mse'])
	#[valinput,valoutput]=load_valdata(divider,canales)
	trainlossactual = model.fit_generator(TrainGen(divider, canales,batch_size,lengthdataset,path,img_y,img_x), callbacks=[check],steps_per_epoch=math.floor(lengthdataset / batch_size),validation_data=load_valdata(divider, canales,batch_size,lengthdataset,path,img_y,img_x), validation_steps=1000,epochs=epochs, verbose=1)

if (VERSION == 1):

	divider = 2
	canales = 1
	image_input = Input(shape=(int(img_y / divider), int(img_x / divider), 1))
	x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(image_input)
	x = BatchNormalization(axis=3, name='bn_conv1')(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x) #Activation('relu')(x)
	x = MaxPooling2D((3, 3))(x)

	x = encoding_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))

	x = encoding_conv_block(x, 3, [128, 128, 512], stage=3, block='a')

	x = encoding_conv_block(x, 3, [256, 256, 1024], stage=4, block='a')

	x = decoding_conv_block(x, 3, [1024, 1024, 256], stage=5, block='a', strides=(1, 1))

	x = decoding_conv_block(x, 3, [512, 512, 128], stage=6, block='a')

	x = decoding_conv_block(x, 3, [256, 256, 64], stage=7, block='a')
	x = Cropping2D(cropping=((0, 0), (0, 1)), data_format=None)(x)

	x = UpSampling2D(size=(3, 3))(x)
	x = Conv2DTranspose(64, (7, 7), strides=(2, 2), padding='same', name='co')(x)
	x = Cropping2D(cropping=((0, 0), (1, 1)), data_format=None)(x)
	x = BatchNormalization(axis=3, name='bn_c1')(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	x = Conv2DTranspose(1, (3, 3), padding='same', name='c8o')(x)
	#x = Cropping2D(cropping=((0, 0), (1, 0)), data_format=None)(x)
	x = Activation('sigmoid')(x)
	#x is output
	x2=tensorflow.keras.backend.concatenate([x,image_input],axis=-1) #Output
	refinement1 = fastrefunit(divider, canales + 1,img_y,img_x)
	x2 = refinement1(x2) #x2 is refined output
	x2 = ZeroPadding2D(padding=((0, 0), (2, 2)), data_format=None)(x2)
	#add x to x2
	x2 = x + x2 #This is HRB block
	
	model = Model(inputs=image_input, outputs=[x, x2])
	model.summary()
	check = tensorflow.keras.callbacks.ModelCheckpoint('DPDnet_fast5.h5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='auto', period=1)
	model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=1e-6),loss=['mse', 'mse'])
	trainlossactual = model.fit_generator(TrainGen(divider, canales,batch_size,lengthdataset,path,img_y,img_x), callbacks=[check],steps_per_epoch=math.floor(lengthdataset / batch_size),validation_data=load_valdata(divider, canales,batch_size,lengthdataset,path,img_y,img_x), validation_steps=1000,epochs=epochs, verbose=1)


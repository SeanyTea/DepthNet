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
import os

#VERSION=0# FULL DPDNET
VERSION=1# FAST VERSION
batch_size = 15
epochs=40
aspect_ratio=480/640 #ASPECT RATIO OF KINECT V2
img_x=320#Kinect V2 half width
img_y=round(aspect_ratio*img_x)
path='GOTPD_DATABASE/'
lengthdataset=len(os.listdir(path+'train/imagenes'))

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
	model.load_weights('DPDnet.h5')



	[valinput, valoutput] = test(divider, canales,path,img_x,img_y)
	cv.namedWindow('prediction', cv.WINDOW_NORMAL)
	cv.namedWindow('input', cv.WINDOW_NORMAL)
	cv.namedWindow('output', cv.WINDOW_NORMAL)



	for j in range(1,len(valinput[:,0,0,0]),1):
			thresh=0.3
			predicted=model.predict(valinput[j-1:j,:,:,:], verbose=0,batch_size=1)

			predicted=predicted/np.max(predicted)
			predicted=predicted+(-np.min(predicted))
			#predicted[predicted>thresh]=1
			#predicted[predicted <= thresh] = 0

			cv.imshow('input', valinput[j-1,:,:,0])
			predicted=to_rgb3(predicted[0,:,:,:])
			cv.imshow('prediction',predicted)
			predicted=to_rgb3(valoutput[j-1,:,:,:])
			cv.imshow('output',predicted)
			cv.waitKey(1)
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
	model = Model(inputs=image_input, outputs=[x2])
	model.summary()

	model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),loss=['mse'])
	model.load_weights('DPDnet_fast4.h5')

	[valinput, valoutput] = test(divider, canales,path,img_x,img_y)
	cv.namedWindow('prediction', cv.WINDOW_NORMAL)
	cv.namedWindow('input', cv.WINDOW_NORMAL)
	cv.namedWindow('output', cv.WINDOW_NORMAL)

	for j in range(1, len(valinput[:, 0, 0, 0]), 1):
		thresh = 0.3
		predicted = model.predict(valinput[j - 1:j, :, :, :], verbose=0, batch_size=1)

		predicted = predicted / np.max(predicted)
		predicted = predicted + (-np.min(predicted))
		# predicted[predicted>thresh]=1
		# predicted[predicted <= thresh] = 0

		cv.imshow('input', valinput[j - 1, :, :, 0])
		predicted = to_rgb3(predicted[0, :, :, :])
		cv.imshow('prediction', predicted)
		predicted = to_rgb3(valoutput[j - 1, :, :, :])
		cv.imshow('output', predicted)
		cv.waitKey(1)











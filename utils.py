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
import random as rand
import scipy
import cv2 as cv
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage
from tensorflow.keras.models import Sequential, load_model,Model
from scipy import *
import imageio
import os
import time


def to_rgb3(im):
    # we can use dstack and an array copy
    # this has to be slow, we create an array with
    # 3x the data we need and truncate afterwards
    im=im*(255/np.max(im))
    return np.asarray(np.dstack((im, im, im)), dtype=np.uint8)
def encoding_identity_block(input_tensor, kernel_size, filters, stage, block):
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = SeparableConv2D(filters1, (1, 1), name=conv_name_base + '2a',kernel_regularizer =tf.keras.regularizers.l1( l=0.01))(input_tensor) # ! Change to spatially separable
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

	x = SeparableConv2D(filters2, kernel_size,
			   padding='same', name=conv_name_base + '2b',kernel_regularizer =tf.keras.regularizers.l1( l=0.01))(x)  # ! Change to spatially separable
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

	x = SeparableConv2D(filters3, (1, 1), name=conv_name_base + '2c',kernel_regularizer =tf.keras.regularizers.l1( l=0.01))(x)  # ! Change to spatially separable
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	return x

def decoding_identity_block(input_tensor, kernel_size, filters, stage, block):

	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2DTranspose(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

	x = Conv2DTranspose(filters2, kernel_size,
			   padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

	x = Conv2DTranspose(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	return x

def encoding_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = SeparableConv2D(filters1, (1, 1), strides=strides,
			   name=conv_name_base + '2a',kernel_regularizer =tf.keras.regularizers.l1( l=0.01))(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

	x = SeparableConv2D(filters2, kernel_size, padding='same',
			   name=conv_name_base + '2b',kernel_regularizer =tf.keras.regularizers.l1( l=0.01))(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

	x = SeparableConv2D(filters3, (1, 1), name=conv_name_base + '2c',kernel_regularizer =tf.keras.regularizers.l1( l=0.01))(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = SeparableConv2D(filters3, (1, 1), strides=strides,
					  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = layers.add([x, shortcut])
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	return x

def decoding_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1

	x=UpSampling2D(size=strides, data_format=None)(input_tensor)
	x = SeparableConv2D(filters1, (1, 1),kernel_regularizer =tf.keras.regularizers.l1( l=0.01))(x)
	x = BatchNormalization(axis=bn_axis)(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

	x = SeparableConv2D(filters2, kernel_size,padding='same',kernel_regularizer =tf.keras.regularizers.l1( l=0.01))(x)
	x = BatchNormalization(axis=bn_axis)(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

	x = SeparableConv2D(filters3, (1,1),kernel_regularizer =tf.keras.regularizers.l1( l=0.01))(x)
	x = BatchNormalization(axis=bn_axis)(x)

	shortcut = UpSampling2D(size=strides, data_format=None)(input_tensor)
	shortcut = SeparableConv2D(filters3, (1, 1) )(shortcut)
	shortcut = BatchNormalization(axis=bn_axis)(shortcut)

	x = layers.add([x, shortcut])
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	return x

def refunit(divider,ch,img_y,img_x):

    image_input = Input(shape=(int(img_y/divider), int(img_x/divider), ch))
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(image_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3))(x)

    x = encoding_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = encoding_conv_block(x, 3, [128, 128, 512], stage=3, block='a')

    x = decoding_conv_block(x, 3, [512, 512, 128], stage=6, block='a')

    x = decoding_conv_block(x, 3, [256, 256, 64], stage=7, block='a')
    #x=ZeroPadding2D(padding=(0,1),data_format=None)(x)

    x = UpSampling2D(size=(3, 3))(x)
    x = Cropping2D(cropping=((2, 2), (2, 2)), data_format=None)(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', name='c8o')(x)
    x = Activation('sigmoid')(x)
    modelo = Model(inputs=image_input, outputs=x)
    modelo.summary()
    return modelo
def fastrefunit(divider,ch,img_y,img_x):

    image_input = Input(shape=(int(img_y/divider), int(img_x/divider), ch))
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(image_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((3, 3))(x)

    x = encoding_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = encoding_conv_block(x, 3, [128, 128, 512], stage=3, block='a')

    x = decoding_conv_block(x, 3, [512, 512, 128], stage=6, block='a')

    x = decoding_conv_block(x, 3, [256, 256, 64], stage=7, block='a')

    x = UpSampling2D(size=(3, 3))(x)
    x = Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', name='c8o')(x)
    x = Activation('sigmoid')(x)
    modelo = Model(inputs=image_input, outputs=x)
    modelo.summary()
    return modelo
def load_valdata(divider,canales,batch_size,lengthdataset,path,img_y,img_x):
    valinput=[]
    valoutput=[]
    multiplier=6
    counter=0
    #l1=["validacion/imagenes/seq-P01-M04-A0002-G00-C00-S0101/image%04d.png"]#,"validation/imagenes/seq-P05-M04-A0001-G03-C00-S0030/image%04d.png","validation/imagenes/seq-P00-M02-A0032-G00-C00-S0037/image%04d.png","validation/imagenes/seq-P00-M02-A0032-G00-C00-S0036/image%04d.png"]
    #l2=["validacion/gaussianas/seq-P01-M04-A0002-G00-C00-S0101/image%04d.png"]#,"validation/gaussianas/seq-P05-M04-A0001-G03-C00-S0030/image%04d.png","validation/gaussianas/seq-P00-M02-A0032-G00-C00-S0037/image%04d.png","validation/gaussianas/seq-P00-M02-A0032-G00-C00-S0036/image%04d.png"]
    l1 = ['VAL_DATA/INPUT/image%05d.png']
    l2 = ['VAL_DATA/OUTPUT/image%05d.png']
    l3=[100]#,509,920,868]
    while 1:
        valinput = []
        valoutput=  []
        for j in range(batch_size*counter+1, batch_size*(counter+1)+1):
             ind=np.uint16(np.random.rand()*0)
             j=np.uint16(np.random.rand()*(l3[ind]-5))+1
             img_path = path+l1[ind] % (j)
             imgc = imageio.imread(img_path)
             imgc = cv.resize(imgc, (int(img_x/divider), int(img_y/divider)))
             xc = image.img_to_array(imgc)
             xc = xc / 65536
             if(canales is 3):
                xc=np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
             valinput.append(xc)

             img_path = path+l2[ind] % (j)
             imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
             xc = image.img_to_array(imgc)
             xc = cv.blur(xc, (3, 3))
             xc = np.expand_dims(xc, axis=2)
             xc = xc / 255
             valoutput.append(xc)
        valinput=np.array(valinput)
        valoutput=np.array(valoutput)
        yield valinput,[valoutput,valoutput]

def TrainGen(divider,canales,batch_size,lengthdataset,path,img_y,img_x):
    counter=0
    while 1:
        X = []
        Y=  []
        for j in range(batch_size*counter+1, batch_size*(counter+1)+1):
            j=math.floor(np.random.rand()*(lengthdataset-5))+1
            img_path = path+"TRAIN_DATA/INPUT/image%05d.png"% (j)#"train/imagenes/image%05d.png" % (j)
            imgc = imageio.imread(img_path)
            imgc = cv.resize(imgc, (int(img_x / divider), int(img_y / divider)))
            xc = image.img_to_array(imgc)
            xc = xc / 65536
            if (canales is 3):
                xc = np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
            X.append(xc)

            img_path = path+"TRAIN_DATA/OUTPUT/image%05d.png"% (j)#"train/gaussianas/image%05d.png" % (j)
            imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
            xc = image.img_to_array(imgc)
            xc = cv.blur(xc, (3, 3))
            xc = np.expand_dims(xc, axis=2)
            xcaux = np.copy(xc)
            xcaux = abs(xcaux - 255)
            xc = xc / 255
            Y.append(xc)
        X = np.array(X)
        Y= np.array(Y)
        counter = counter + 1
        yield X,[Y,Y]

def to_rgb3(im):
    # we can use dstack and an array copy
    # this has to be slow, we create an array with
    # 3x the data we need and truncate afterwards
    im=im*(255/np.max(im))
    return np.asarray(np.dstack((im, im, im)), dtype=np.uint8)
def test(divider,canales,path,img_x,img_y):
    valinput=[]
    valoutput=[]
    multiplier=6
    counter=0
    valinput = []
    valoutput=  []
    for j in range(1,2200,1):
         #img_path = path+"validacion/imagenes/seq-P01-M04-A0002-G00-C00-S0101/image%04d.png" % (j)
         #img_path = path+"frontal/validation/DEPTH_SEQUENCE/image%05d.png"% (j)#"train/imagenes/image%05d.png" % (j)
         img_path = path+"TRAIN_DATA/INPUT/image%05d.png" %(j)
         imgc = imageio.imread(img_path)
         imgc = cv.resize(imgc, (int(img_x / divider), int(img_y / divider)))
         xc = image.img_to_array(imgc)
         xc = xc / 65536
         valinput.append(xc)

         #img_path = path+"validacion/gaussianas/seq-P01-M04-A0002-G00-C00-S0101/image%04d.png" % (j)
         #img_path = path+"frontal/validation/GAUSSIANAS/image%05d.png"% (j)#"train/imagenes/image%05d.png" % (j)
         img_path = path+"TRAIN_DATA/OUTPUT/image%05d.png" %(j)
         imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
         xc = image.img_to_array(imgc)
         xc = cv.blur(xc, (3, 3))
         xc = np.expand_dims(xc, axis=2)
         xc = xc / 255
         valoutput.append(xc)
    valinput=np.array(valinput)
    valoutput=np.array(valoutput)
    return valinput,valoutput







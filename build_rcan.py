import os, glob, re, signal, sys, argparse, threading, time, h5py, math, random
import scipy.misc 
import scipy.io
from skimage.measure import compare_ssim
from random import shuffle
from PIL import Image
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from config import conv_layer

# def conv_layer(x,ch,filter_size,use_bias,name,reuse,paddingMode='same'):
#     x = tf.layers.conv2d(x,ch,filter_size,padding=paddingMode,use_bias = use_bias,
#                 kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch/ch/ch/ch)),
#                 bias_initializer=tf.zeros_initializer(),
#                 activation=None,
#                 name=name,
#                 reuse=reuse)
#     return x

def res_channel_attention_block(x,name,reuse):
    skip = x
    x = conv_layer( x,64,3,True,name+'conv_01',reuse)
    x = tf.nn.relu( x )
    x = conv_layer( x,64,3,True,name+'conv_02',reuse)

    # channel attention
    ca = tf.reduce_mean( x, axis=[1,2],keepdims=True)
    ca = conv_layer(ca, 4,1,True,name+'ca_conv_01',reuse)
    ca = tf.nn.relu( ca )
    ca = conv_layer( ca, 64,1,True,name + 'ca_conv_02',reuse)
    ca = tf.nn.sigmoid( ca )

    x = x * ca
    y = x + 0.1*skip
    return y

def residual_group( x,name,reuse ):
    skip = x
    for i in range(10):
        x = res_channel_attention_block(x,name+'rcab_%02d'%(i),reuse)
    y = x + 0.1*skip
    return y

def build_rcan( x, reuse,ch=1):
    skip = x
    net = []
    x = conv_layer( x,64,3,True,'conv_01',reuse)
    x = tf.nn.relu( x)
    net.append( x )
    for i in range(5):
        x = residual_group( x, 'RCAN_group_%02d'%(i),reuse) # contains 10 res_attention_block with 2 convs, 20 convs in total
        net.append( x )

    x = conv_layer( x,ch,3,True,'conv_final',reuse)
    y = x + 0.1*skip
    return y,net

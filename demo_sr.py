import os, glob, re, signal, sys, argparse, threading, time, math, random
import scipy.misc 
import scipy.io
from skimage.measure import compare_ssim
from random import shuffle
from PIL import Image
# from nlrn_nobn import build_nlrn

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils_py import *
import class_realsr as model
from utils_degrade import *


def parse_args():
    parser = argparse.ArgumentParser(description='real sr arguments')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--phase',type=str,default='test', help='train or test or degrade')
    parser.add_argument('--data',type=str,default='syn_real', help='real or syn or syn_real')
    parser.add_argument('--hrset',type=str,default='d', help='diffract or wesaturate or flickr2k')
    parser.add_argument('--mode',type=str,default='rcan', help='vdsr or rcan')
    parser.add_argument('--sigma',type=int,default=7, help='AWGB noise sigma')
    parser.add_argument('--epoch',dest='MAX_EPOCH',type=int,default=3000)
    parser.add_argument('--scale',dest='factor',type=str,default='3')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.chdir('..')
    print(os.getcwd())
    
    # set params
    args = parse_args()

    args.IMG_SIZE = (192, 192)
    args.BATCH_SIZE = 16
    args.BASE_LR = 0.0001
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # set up realsr models
    realsr = model.realsr(args)
    model_name = 'r_k15n8_' 
    # model_name = 'kmsr' 

    # global setting
    realsr.num_k_basis = 1
    realsr.k_size = 15
    realsr.ks_group = [7,5,5]
    if args.phase == 'degrade':

        if model_name == 'uniform':
            realsr.degrade_model_name = 'uniform' 
            realsr.degrade_EPOCH = 60
            realsr.uniform_degrade_net()
        elif model_name == 'degradeNet':
            realsr.degrade_model_name = 'dnet' 
            realsr.degrade_model_name = 'kpn' 
            realsr.degrade_EPOCH = 60
            realsr.degrade_net()
        else:
            realsr.degrade_model_name = model_name 
            realsr.degrade_EPOCH = 60
            ## debugging
            realsr.degradation()
        
        

    elif args.phase == 'train':
        # train and evaluate
        realsr.tag = model_name
        realsr.BATCH_SIZE = 16
        if realsr.mode == 'rcan':
            realsr.BATCH_SIZE = 16
            realsr.IMG_SIZE   = (80,80)
            realsr.MAX_EPOCH  = 8000
        realsr.train( model_name )

    elif args.phase == 'test':
        # test only
        realsr.reuse = False
        realsr.test( model_name )
        '''
        tag = '00004'
        test_path = '.\\realsr_database\\LearnToZoom\\rgbx3'
        out_path = os.path.join('.\\realsr_database\\LearnToZoom\\sr_result',tag)
        # test_path = '.\\test_vis\\LearnToZoom'
        out_path = '.\\test_vis\\LearnToZoom\\x3\\rgb'
        # test_path = 'E://sr_test//test_imgs'
        # out_path  = 'E://sr_test'
        realsr.test_vis_result( model_name ,test_path,out_path,tag,mode = 'rgb')
        '''
    elif args.phase == 'syn_test':
        realsr.reuse = False
        realsr.dataset = 'BSD100'
        realsr.syn_test( model_name )

import numpy as np
import math,os,glob,random
from scipy import signal
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt 
import cv2
import scipy.misc 
import scipy.stats as ss

def save_png(name,output):
    scipy.misc.imsave(name,np.clip(output*255,0,255).astype( 'uint8'))
    return 0

def check_checkpoints(path):
    model_lists = sorted(glob.glob(os.path.join(path,"*epoch_*.meta")))
    if model_lists:
        sort_model_lists = []
        epoches = []
        for i,model_name in enumerate(model_lists):
            ckpt_name  = model_name.split('.meta')[0]
            sort_model_lists.append(ckpt_name)
            epoches.append(int(ckpt_name.split('epoch_')[1].split('.ckpt')[0]))
        model_lists = [x for _, x in sorted(zip(epoches,sort_model_lists))]
        start = np.array(epoches).max() + 1
        ckpt_name = model_lists[-1]
        print( "restore model from epoch %d\t"%(start))
    else:
        start = 0
        ckpt_name = []
    return  ckpt_name,start,model_lists


def test_patch(test_img, input_tensor, output_tensor,sess,prob,test_sze=256,stride=144,mode='gray'):
    h,w = test_img.shape[:2]
    if mode =='gray':
        output = np.zeros((h,w))
        weight = np.zeros((h,w))
    else:
        output = np.zeros((h,w,3))
        weight = np.zeros((h,w,3))
    x_s = list(range(0,h-test_sze,stride));x_s.append(h-test_sze)
    y_s = list(range(0,w-test_sze,stride));y_s.append(w-test_sze)
    for x in x_s:
        for y in y_s:
            if mode =='gray':
                recover = sess.run(output_tensor,
                        feed_dict={input_tensor: test_img[np.newaxis,x:x+test_sze,y:y+test_sze,np.newaxis],
                        prob:1.0})
                output[x:x+test_sze,y:y+test_sze] += np.squeeze(recover)
                weight[x:x+test_sze,y:y+test_sze] += 1
            else:
                recover = sess.run(output_tensor,
                        feed_dict={input_tensor: test_img[np.newaxis,x:x+test_sze,y:y+test_sze,:],
                        prob:1.0})
                output[x:x+test_sze,y:y+test_sze,:] += np.squeeze(recover)
                weight[x:x+test_sze,y:y+test_sze,:] += 1
    output = output / weight
    return output 

def test_sr_patch(test_img, input_tensor, output_tensor,sess,prob,scale=2,test_sze=256,stride=144):
    in_h,in_w = test_img.shape[:2]
    out_h = in_h*scale
    out_w = in_w*scale
    output = np.zeros((out_h,out_w,3))
    weight = np.zeros((out_h,out_w,3))
    x_s = list(range(0,in_h-test_sze,stride));x_s.append(in_h-test_sze)
    y_s = list(range(0,in_w-test_sze,stride));y_s.append(in_w-test_sze)
    for x in x_s:
        for y in y_s:
            recover = sess.run(output_tensor,
                        feed_dict={input_tensor: test_img[np.newaxis,x:x+test_sze,y:y+test_sze,:],
                        prob:1.0})
            output[x*scale:x*scale+test_sze*scale,y*scale:y*scale+test_sze*scale,:] += np.squeeze(recover)
            weight[x*scale:x*scale+test_sze*scale,y*scale:y*scale+test_sze*scale,:] += 1
    output = output / weight
    return output

def test_kpn_degrade_patch(test_img, input_tensor, output_tensor,sess,prob,scale=2,test_sze=256,stride=144):
    in_h,in_w = test_img.shape[:2]
    out_h = in_h//scale
    out_w = in_w//scale
    output = np.zeros((out_h,out_w,3))
    weight = np.zeros((out_h,out_w,3))
    lr_sze = test_sze // scale
    x_s = list(range(0,in_h-test_sze,stride));x_s.append(in_h-test_sze)
    y_s = list(range(0,in_w-test_sze,stride));y_s.append(in_w-test_sze)
    for x in x_s:
        for y in y_s:
            recover = sess.run(output_tensor,
                        feed_dict={input_tensor: test_img[np.newaxis,x:x+test_sze,y:y+test_sze,:],
                        prob:1.0})
            output[x//scale:x//scale+lr_sze,y//scale:y//scale+lr_sze,:] += np.squeeze(recover)
            weight[x//scale:x//scale+lr_sze,y//scale:y//scale+lr_sze,:] += 1
    output = output / weight
    return output 

def im2patches(img,sze,stride):
    h,w = img.shape[:2]
    h_range = list(range(0,h-sze,stride))
    h_range.append(h-sze)
    w_range = list(range(0,w-sze,stride))
    w_range.append(w-sze)

    patches = []
    for x in h_range:
        for y in w_range:
            if len(img.shape) == 2:
                patches.append(img[x:x+sze,y:y+sze])
            elif len(img.shape) ==3:
                patches.append(img[x:x+sze,y:y+sze,:])
    return patches


    
def get_model_lists(path):
    model_list = sorted(glob.glob(os.path.join(path,"*epoch_*")))
    model_list = [fn for fn in model_list if os.path.basename(fn).endswith("meta")]
    return model_list

def read_img(filename):
    if os.path.exists( filename ):
        im = scipy.misc.imread(filename).astype('float32')/255
    else:
        im = []
        print('!!! error !!! empty image',filename)
    return im

def read_img_y(filename):
    img = read_img(filename)
    img_ycc = rgb2ycbcr_matlab(img)
    return img_ycc[:,:,0]

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = g/g.sum()
    return g.astype('float32')

def anistropic_gaussian_v2(sze, s1, s2):
    x, y = np.mgrid[-sze//2 + 1:sze//2 + 1, -sze//2 + 1:sze//2 + 1]
    z = np.array(range(-sze//2 + 1,sze//2 + 1))
    g1 = np.exp(-((z**2 )/(2.0*s1**2)))
    g2 = np.exp(-((z**2 )/(2.0*s2**2)))
    # g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    # kernel = [exp(-x**2/(2*s**2)) for z in range(-k,k+1)] 
    kernel = np.outer(g1,g2.T)
    kernel = kernel / kernel.sum()
    return kernel
    
def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return 0 

def data_augment( tensor_lists):
    num_inputs = len(tensor_lists)
    z = tensor_lists[0]
    a = random.random()
    if a < 0.15:  # up down flip
        output = map(lambda x:x[:,::-1,:,:],tensor_lists) if num_inputs >1 else z[:,::-1,:,:]
    elif a < 0.3: # left right flip
        output = map(lambda x:x[:,:,::-1,:],tensor_lists) if num_inputs >1 else z[:,:,::-1,:]
    elif a < 0.45: # transpose image
        output = map(lambda x:x.transpose((0,2,1,3)),tensor_lists) if num_inputs >1 else z.transpose((0,2,1,3))
    elif a < 0.6:
        output = map(lambda x:np.rot90(x,k=1,axes=(1,2)),tensor_lists) if num_inputs >1 else np.rot90(z,k=1,axes=(1,2))
    elif a <0.75:
        output = map(lambda x:np.rot90(x,k=2,axes=(1,2)),tensor_lists) if num_inputs >1 else np.rot90(z,k=2,axes=(1,2))
    elif a < 0.9:
        output = map(lambda x:np.rot90(x,k=3,axes=(1,2)),tensor_lists) if num_inputs >1 else np.rot90(z,k=3,axes=(1,2))
    else:
        output = tensor_lists if num_inputs >1 else z
    return output

def conv_layer(x,ch,filter_size,use_bias,name,reuse,paddingMode='same',strides=1):
    x = tf.layers.conv2d(x,ch,filter_size,padding=paddingMode,use_bias = use_bias,strides=(strides,strides),
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch/2)),
                    bias_initializer=tf.zeros_initializer(),activation=None,name=name,reuse=reuse)
    return x



def stack_convs(x,ch,layers,reuse,name='conv',mode='conv_first',fs=3):
    for i_level in range(layers):
        if mode == 'conv_first':
            x = conv_layer( x,ch,filter_size=fs,use_bias = True,name=name+'_%02d'%(i_level),reuse=reuse)
            x = tf.nn.relu(x)
        elif mode == 'relu_first':
            x = tf.nn.relu(x)
            x = conv_layer( x,ch,filter_size=fs,use_bias = True,name=name+'_%02d'%(i_level),reuse=reuse)
    return x

def stack_resblock(x,ch,layers,reuse,scope):
    x = conv_layer( x,ch,filter_size=3,use_bias = True,name='conv0',reuse=reuse)
    with tf.variable_scope(scope, reuse=reuse):
        if layers < 5:
            for i_level in range(layers):
                x = res_block(x,ch,3,'res%02d'%(i_level),reuse)
        else:
            for i_short in range(layers//5):
                short_skip = x
                for i_level in range(5):
                    x = res_block(x,ch,3,'res%02d_%02d'%(i_short,i_level),reuse)
                x = x + short_skip
    return x 

def extract_feature_unet(x, reuse,scope,layers=10,ch=32):
    x = conv_layer( x,ch,filter_size=3,use_bias = True,name='conv0',reuse=reuse)
    with tf.variable_scope(scope, reuse=reuse):
        x = stack_convs(x,ch,3,reuse,name='enc',mode='relu_first')
        concat = x 
        x = tf.nn.max_pool( x,[1,2,2,1],[1,2,2,1],padding='SAME')  
        x = stack_convs(x,ch,3,reuse,name='conv',mode='relu_first')
        x = tf.image.resize_bilinear(x, [tf.shape(x)[1]*2,tf.shape(x)[2]*2])
        x = tf.concat((x,concat),axis=-1)
        x = stack_convs(x,ch,3,reuse,name='dec')

    return x

def res_block( x,ch,ks,name,reuse):
    skip = x 
    x = tf.nn.leaky_relu(x,0.2)
    x = conv_layer(x,ch,ks,True,name+'conv_01',reuse)
    x = tf.nn.leaky_relu(x)
    x = conv_layer(x,ch,ks,True,name+'conv_02',reuse)
    x = x + skip
    return x 

def cal_psnr(target, ref, border =0, max_value=1.0):
    #assume RGB image
    target_data = np.array(target)
    ref_data = np.array(ref)
    
    diff = ref_data - target_data
    if border:
        diff = diff[border:-border,border:-border]

    diff = diff.flatten('C')  #适用于numpy对象,返回一个折叠成一维的数组,'C'表示row-major
    rmse = math.sqrt( np.mean(diff ** 2.) )
    psnr = 20*math.log10(max_value/rmse)
    return psnr


    
def anistropic_gaussian(ksize=15, theta=np.pi, l1=6, l2=6):

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)
    k = k.astype('float32')
    return k

def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k

def rgb2ycbcr_matlab(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def ycbcr2rgb_matlab(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*255.0-16)/(235-16) #to [0, 1]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*255.0-16)/(240-16) #to [0, 1]
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb


def shuffle_up( x,channel,scale):
    # channels first, then subpixel  into |1|2|
    #                                     |3|4|
    bs,h,w,c = get_tensor_NHWC(x)
    y = tf.transpose(x,(0,2,1,3))
    y = tf.reshape(y,(bs,w,h*scale,channel*scale))
    y = tf.transpose(y,(0,2,1,3))
    y = tf.reshape(y,(bs,h*scale,w*scale,channel))
    return y

def shuffle_down(x,channel,scale):
    # method |1|2| channels first, then subpixel
    #        |3|4|
    bs,h,w,c = get_tensor_NHWC(x)
    y = tf.reshape(x,(bs,h,-1))
    y = tf.reshape(y,(bs,h,w//scale,c*scale))
    y = tf.transpose(y,(0,2,1,3))
    y = tf.reshape(y,(bs,w//scale,-1))
    y = tf.reshape(y,(bs,w//scale,h//scale,c*scale**2))
    y = tf.transpose(y,(0,2,1,3))
    return y

def get_tensor_NHWC(tensor):
    sze = tf.shape(tensor)
    return sze[0],sze[1],sze[2],sze[3]

def imshow( im):
    if len(im.shape) > 2:
        plt.imshow(im);plt.show()
    else:
        plt.imshow(im,'gray');plt.show()
    return 0

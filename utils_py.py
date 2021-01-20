import numpy as np
import math,os,glob
from scipy import signal
from scipy import ndimage
from scipy.ndimage.interpolation import rotate
import tensorflow as tf
import matplotlib.pyplot as plt
# import vgg16
from config import extract_feature_unet, conv_layer# ,kpn_module
import cv2, random
import scipy.misc 

class ImageRecord:
  def __init__(self, label,img22,img18,img14):
    self.label = label
    self.img22 = img22
    self.img18 = img18
    self.img14 = img14

def im_to_patch( im, sze, stride):
    patches = []
    for x in range(0,im.shape[0]-sze-1,stride):
        for y in range(0,im.shape[1]-sze-1,stride):
            patches.append( im[x:x+sze,y:y+sze,:])
    return patches


def crop_patch( filename ,sze, stride ,border=0):
    if os.path.exists( filename ):
        im = scipy.misc.imread(filename).astype('float32')/255
        if border:
            im = im[border:-border,border:-border,:]
        patches = im_to_patch( im, sze, stride)
    else:
        patches = []
    return patches

def get_train_lists( imdb, aperture_size):
    train_lists = {}
    train_lists['22'] = [i for i in range(len(imdb)) if imdb[i].img22 != []]
    train_lists['18'] = [i for i in range(len(imdb)) if imdb[i].img18 != []]
    train_lists['14'] = [i for i in range(len(imdb)) if imdb[i].img14 != []]
    return train_lists[aperture_size]




def load_imdb( imdb_path, camera ):
    folder = os.path.join( './/database', camera , 'tif_reg//train')
    imdb = []
    labels = sorted( glob.glob(os.path.join(folder, '*_7.1_reg.png')))

    sze = 128
    stride =  90

    for i_fold in range(len(labels)):
        group_name = labels[i_fold].split('_7.1')[0][-6:]
        print( 'processing training image No.',group_name)
        # group_name = '000014'
        label_name = os.path.join( folder, group_name + '_7.1_reg.png')
        label  = crop_patch( label_name,sze,stride)

        f22_name = os.path.join( folder, group_name + '_22.0_reg_c.png')
        im_f22 = crop_patch( f22_name ,sze,stride)

        f18_name = os.path.join( folder, group_name + '_18.0_reg_c.png')
        im_f18 = crop_patch( f18_name ,sze,stride)

        f14_name = os.path.join( folder, group_name + '_14.0_reg_c.png')
        im_f14 = crop_patch( f14_name ,sze,stride)


        for i_patch in range(len(label)):
            label_patch = label[i_patch]
            im_f22_patch = im_f22[i_patch] if im_f22 else []
            im_f18_patch = im_f18[i_patch] if im_f18 else []
            im_f14_patch = im_f14[i_patch] if im_f14 else []
            imdb.append( ImageRecord(
                    label=label_patch,
                    img22=im_f22_patch,
                    img18=im_f18_patch,
                    img14=im_f14_patch)) 
        del label, im_f22, im_f18, im_f14

    return imdb



def zero_shot_imdb(  camera,i_fold ):
    folder = os.path.join( './/database', camera , 'tif_reg//train')
    imdb = []
    labels = sorted( glob.glob(os.path.join(folder, '*_7.1_reg.png')))

    sze = 128
    stride =  90

    
    group_name = labels[i_fold].split('_7.1')[0][-6:]
    print( 'processing image No.',group_name)
        
    label_name = os.path.join( folder, group_name + '_7.1_reg.png')
    label_patch = read_img(label_name)

    f22_name = os.path.join( folder, group_name + '_22.0_reg_c.png')
    im_f22 = read_img( f22_name )

    f18_name = os.path.join( folder, group_name + '_18.0_reg_c.png')
    im_f18 = read_img( f18_name )

    f14_name = os.path.join( folder, group_name + '_14.0_reg_c.png')
    im_f14 = read_img( f14_name )

    imdb.append( ImageRecord(
            label=label_patch,
            img22=im_f22,
            img18=im_f18,
            img14=im_f14)) 
    
    return imdb


def em(tensor, input_sze, h_K, w_K, T):
    mu = []
    z  = []
    tensor_reshape = tf.reshape( tensor, [tf.shape(tensor)[0],
                                            tf.shape(tensor)[1] * tf.shape(tensor)[2],
                                            tf.shape(tensor)[3]  ])
    # initiate mu_0, of size BS*K*C
    step_h = input_sze[0]// h_K
    step_w = input_sze[1]// w_K
    
    mu_0 = tensor[:,step_h//2:(step_h//2+step_h*h_K):step_h,step_w//2:(step_w//2+step_w*w_K):step_w,:] # slice from x_phi to obtain mu_0 of size Bs*H*W*K
    mu_0 = tf.reshape( mu_0,[tf.shape(mu_0)[0],h_K*w_K,tf.shape(mu_0)[-1]])
    mu_0 = tf.transpose(mu_0,[0,2,1])
    mu_0 = tf.stop_gradient(mu_0)
    mu.append(mu_0)

    # Z_1 = tf.nn.softmax( tf.nn.conv2d(x_phi, mu_0, strides=[1,1,1,1], padding='SAME') ,axis=-1) # size:BS*H*W*K
    for time in range(T):
        # update Z of size BS*HW*K
        mu_temp = mu[time]
        z_temp = tf.nn.softmax( tf.matmul( tensor_reshape, mu_temp ))
        # z_temp = tf.nn.softmax( tf.nn.conv2d(x_phi, mu_0, strides=[1,1,1,1], padding='SAME') ,axis=-1) # size:BS*H*W*K
        z.append(z_temp)

        # updata mu
        mu_temp = tf.matmul( tf.transpose( tensor_reshape,[0,2,1]), z_temp )
        norm    = tf.reduce_sum( z_temp, axis = 1,keepdims = True) # norm of size BS*1*K
        mu_temp = tf.math.divide( mu_temp, norm)
        mu.append(mu_temp)
    return z,mu

# def perceptual_loss(train_output,target):
#     vgg_s = vgg16.Vgg16('.//vgg//vgg16.npy')
#     vgg_s.build(target)
#     feature = vgg_s.conv2_2
#     # feature = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3]

#     vgg_s.build(train_output)
#     feature_ = vgg_s.conv2_2
    
#     loss = tf.reduce_mean(tf.subtract(feature, feature_) ** 2, [0,1, 2, 3])
#     return loss

def side_conv_layer(x,ch,filter_size,use_bias,name,reuse,batchsize=64,inH=48,inW=48):
    pad_num = int(filter_size-1)
    pad_I   = tf.pad( x,[[0,0],[pad_num,0],[0,pad_num],[0,0]],'SYMMETRIC')
    pad_II  = tf.pad( x,[[0,0],[pad_num,0],[pad_num,0],[0,0]],'SYMMETRIC')
    pad_III = tf.pad( x,[[0,0],[0,pad_num],[pad_num,0],[0,0]],'SYMMETRIC')
    pad_IV  = tf.pad( x,[[0,0],[0,pad_num],[0,pad_num],[0,0]],'SYMMETRIC')

    feat_I  = tf.layers.conv2d(pad_I,ch,filter_size,padding='valid',use_bias = use_bias,
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch)),
                    bias_initializer=tf.zeros_initializer(),activation=None,name=name+'_ru',reuse=reuse)
    feat_II  = tf.layers.conv2d(pad_II,ch,filter_size,padding='valid',use_bias = use_bias,
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch)),
                    bias_initializer=tf.zeros_initializer(),activation=None,name=name+'_lu',reuse=reuse)
    feat_III  = tf.layers.conv2d(pad_III,ch,filter_size,padding='valid',use_bias = use_bias,
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch)),
                    bias_initializer=tf.zeros_initializer(),activation=None,name=name+'_lb',reuse=reuse)
    feat_IV  = tf.layers.conv2d(pad_IV,ch,filter_size,padding='valid',use_bias = use_bias,
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch)),
                    bias_initializer=tf.zeros_initializer(),activation=None,name=name+'_rb',reuse=reuse)
    feat  = tf.concat([tf.expand_dims(feat_I,-1),tf.expand_dims(feat_II,-1),
                    tf.expand_dims(feat_III,-1),tf.expand_dims(feat_IV,-1)],axis = -1)
    distance = tf.reduce_sum( tf.squared_difference( feat, tf.tile(tf.expand_dims(x,-1),(1,1,1,1,4))),  axis = -2)
    idx   = tf.argmin( distance, axis = -1)

    # compute index
    w, h = tf.meshgrid(tf.range(inW), tf.range(inH)) # w,h = tf.meshgrid(w, h)
    batch_idx = tf.reshape(tf.range(batchsize), (batchsize, 1, 1))
    b = tf.tile(batch_idx, (1, inH, inW))
    h = tf.tile( tf.expand_dims(h,0),(batchsize,1,1))
    w = tf.tile( tf.expand_dims(w,0),(batchsize,1,1))
    feat_idx = tf.stack([b,h,w,tf.to_int32(idx)], axis=-1)


    sd_feat  = tf.gather_nd( tf.transpose(feat,perm=[0,1,2,4,3]) , feat_idx)
    return sd_feat


def side_conv_max_layer(x,ch,filter_size,use_bias,name,reuse,paddingMode='same'):
    pad_num = int(filter_size-1)
    pad_I   = tf.pad( x,[[0,0],[pad_num,0],[0,pad_num],[0,0]],'SYMMETRIC')
    pad_II  = tf.pad( x,[[0,0],[pad_num,0],[pad_num,0],[0,0]],'SYMMETRIC')
    pad_III = tf.pad( x,[[0,0],[0,pad_num],[pad_num,0],[0,0]],'SYMMETRIC')
    pad_IV  = tf.pad( x,[[0,0],[0,pad_num],[0,pad_num],[0,0]],'SYMMETRIC')
    # compute side window feature, 4 groups of size N*H*W*C
    feat_I  = tf.layers.conv2d(pad_I,ch,filter_size,padding='valid',use_bias = use_bias,
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch)),
                    bias_initializer=tf.zeros_initializer(),activation=None,name=name+'_ru',reuse=reuse)
    feat_II  = tf.layers.conv2d(pad_II,ch,filter_size,padding='valid',use_bias = use_bias,
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch)),
                    bias_initializer=tf.zeros_initializer(),activation=None,name=name+'_lu',reuse=reuse)
    feat_III  = tf.layers.conv2d(pad_III,ch,filter_size,padding='valid',use_bias = use_bias,
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch)),
                    bias_initializer=tf.zeros_initializer(),activation=None,name=name+'_lb',reuse=reuse)
    feat_IV  = tf.layers.conv2d(pad_IV,ch,filter_size,padding='valid',use_bias = use_bias,
                    kernel_initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(filter_size**2)/ch)),
                    bias_initializer=tf.zeros_initializer(),activation=None,name=name+'_rb',reuse=reuse)
    feat  = tf.concat([tf.expand_dims(feat_I,-1),tf.expand_dims(feat_II,-1),
                    tf.expand_dims(feat_III,-1),tf.expand_dims(feat_IV,-1)],axis = -1)
    feat  = tf.reduce_max(feat,axis = -1)
    return feat

def kpn_module(input_tensor,per_pixel_weight,ks=3,bs=64,channels=3):
    bs = tf.shape(input_tensor)[0]
    h  = tf.shape(input_tensor)[1]
    w  = tf.shape(input_tensor)[2]
    pad_num = int( (ks-1)/2 )
    x = tf.pad( input_tensor,[[0,0],[pad_num,pad_num],[pad_num,pad_num],[0,0]], mode='symmetric')
    input_matrix = tf.image.extract_image_patches(x, [1,ks,ks,1],[1,1,1,1],[1,1,1,1],"VALID")

    kernel = tf.reshape( per_pixel_weight,(bs,h,w,ks*ks,channels)) # kernel of size:bs*hw*ks^2*ch
    input_matrix  = tf.reshape( input_matrix, (bs,h,w,ks*ks,channels))
    result = tf.multiply(kernel, input_matrix) # size bs*(hw)*(ks*ks)*chanel
    result = tf.reduce_sum( result, -2) # size bs*(hw)*1*channels
    # tensor = tf.reshape( result, (bs, height, width,channels))
    return result


def tf_gaussian_kernel(size: int,mean: float,std: float,):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij',vals,vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)



def LoG(size = 7, sigma = 0.6):
    '''
    kernel = np.array([[ 0,0,1,0,0],
                       [ 0,1,2,1,0],
                       [ 1,2,-16,2,1],
                       [0,1,2,1,0],
                       [0,0,1,0,0]],
                       dtype=np.float32)
    '''
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    t = (x**2 + y**2)/(2.0*sigma**2)
    g = - (1/np.pi/sigma**4)*(1-t) * np.exp(-t)
    g = g/g.sum()
    return g.astype('float32')


def comp_filter_loss( residual ,kernel,name='filter'):
    kernel = tf.tile( tf.expand_dims(tf.expand_dims(tf.constant(kernel, name=name ),-1),-1),(1,1,3,1))
    loss = tf.nn.depthwise_conv2d(residual, kernel, [1, 1, 1, 1], padding='SAME')
    loss = tf.reduce_sum( tf.nn.l2_loss(loss) )
    # loss = tf.reduce_sum( tf.math.abs( loss ))
    return loss

def DoG():
    dog = gkernel(sigma=0.95) - gkernel(sigma=2.75)
    return dog

def gkernel(size = 11, sigma = 0.95):
    from cv2 import getGaussianKernel, CV_32F
    mat = getGaussianKernel(size, sigma, CV_32F)
    return np.dot(mat, np.transpose(mat))

def lap_pyr(im,level):
    layer = im
    gaussian_pyramid = [im]
    for i in range(level):
        layer = cv2.pyrDown(layer)
        gaussian_pyramid.append(layer)
    # Laplacian Pyramid
    layer = gaussian_pyramid[level]
    laplacian_pyramid = [layer]
    for i in range(level, 0, -1):
        size = (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    # reconstruction, can reconstruct the original image, i.e., reconstructed image = im
    # reconstructed_image = laplacian_pyramid[0]
    # for i in range(1, level+1):
    #     size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
    #     reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
    #     reconstructed_image = cv2.add(reconstructed_image, laplacian_pyramid[i])
    return laplacian_pyramid

    


def sobel_psnr( target , ref, thresh=0.1):
    pad_ref = np.pad( ref, ((1,1),(1,1),(0,0)),'constant')
    
    grad_x = pad_ref[:-2,:-2,:]+2*pad_ref[1:-1,:-2,:] + pad_ref[2:,:-2,:] - (pad_ref[:-2,2:,:]+2*pad_ref[1:-1,2:,:] + pad_ref[2:,2:,:] )
    grad_y = pad_ref[:-2,:-2,:]+2*pad_ref[:-2,1:-1,:] + pad_ref[:-2,2:,:] - (pad_ref[2:,:-2,:]+2*pad_ref[2:,1:-1,:] + pad_ref[2:,2:,:] )
    grad = np.abs( grad_x )+ np.abs( grad_y)
    mask = ( grad >= thresh)
    mask = np.max( mask,2,keepdims=True)
    mask = np.tile( mask, (1,1,3))

    psnr = cal_psnr(target[mask] ,ref[mask])
    diff = target[mask] - ref[mask]
    l2 = math.sqrt( np.mean(diff ** 2.) )
    l1 = np.mean( np.abs(diff) )
    return psnr , l2, l1, mask

def im2col(image,sze,stride,channels):
    h = image.shape[1]
    w = image.shape[2]
    s_h = math.ceil((h-sze)/stride+1)
    s_w = math.ceil((w-sze)/stride+1)
    patches = np.zeros([s_h,s_w,sze,sze,channels],dtype='float32')
    i_vertical = list(range(0,h-sze,stride))
    i_horizon = list(range(0,w-sze,stride))
    i_vertical.append(h-sze)
    i_horizon.append(w-sze)

    for i_v in range(len(i_vertical)):
        for i_h in range(len(i_horizon)):
            s_v = i_vertical[i_v]
            s_h = i_horizon[i_h]
            patches[i_v,i_h,:,:,:] = image[:,s_v:s_v+sze,s_h:s_h+sze,:]
    return patches

def col2im( sze,stride,h,w,output_p,channels, border):
    i_vertical = list(range(0,h-sze,stride))
    i_horizon = list(range(0,w-sze,stride))
    i_vertical.append(h-sze)
    i_horizon.append(w-sze)

    im = np.zeros((h,w,channels),dtype='float32')
    weight  = np.zeros((h,w,channels),dtype='float32')
    border = border
    for i_v in range( len(i_vertical) ):
        for i_h in range( len(i_horizon) ):
            s_v = i_vertical[i_v]
            s_h = i_horizon[i_h]
            im[s_v+border:(s_v+sze-border),s_h+border:(s_h+sze-border),: ] += output_p[i_v,i_h,:,:,:]                                
            weight[s_v+border:(s_v+sze-border),s_h+border:(s_h+sze-border),: ] += 1
    output =  im/weight
    
    return output

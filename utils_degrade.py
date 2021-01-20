import numpy as np
import math,os,glob,random,time
from scipy import signal
from scipy import ndimage
import tensorflow as tf
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt 
import cv2
import scipy.misc 
from config import *


def get_sr_test_lists(factor,folder='.//realsr_database//RealSR_ycc'):
    folder = [os.path.join(folder,'Canon//Test'), os.path.join(folder,'Nikon//Test')]
    folder = [os.path.join(i,factor) for i in folder]
    test_lists = []
    hr_lists = glob.glob(os.path.join(folder[0],'*HR.png')) 
    hr_lists += glob.glob(os.path.join(folder[1],'*HR.png'))
    test_lists = [[i,i.replace('HR','LR'+factor)] for i in hr_lists]
    return test_lists 

def get_div2k_syn_data_online(factor,tag,N,k_size,deep_k_group,border,hrset='all',mode='v2'):
    # def degradation graph
    model_path  = os.path.join('.//sr_degrade', 'checkpoints_' + 'x%s_'%factor + tag ) 
    ckpt_name,_,_ = check_checkpoints(model_path)
    train_input = tf.placeholder(tf.float32, shape=(None,None,None, 1))
    prob   = tf.placeholder(tf.float32)
    coefs = build_degrade_net(train_input,N,1,reuse = False,scope = 'degrade') 

    if 'deep_' in tag :
        k_group  = [tf.get_variable('k%02d'%i,[deep_k_group[i],deep_k_group[i],1,N],
                        initializer=tf.random_normal_initializer) for i in range(len(deep_k_group))]
        train_output,_ = apply_deep_kpn_basis(train_input,k_group,deep_k_group,k_size,coefs)
    else:
        kernel_basis = tf.get_variable('k_basis',[k_size,k_size,1,N],tf.random_normal_initializer)
        train_output,_ = apply_kpn_basis(train_input,kernel_basis,k_size,coefs)

    if hrset == 'd':    hrsets  = ['diffract']
    if hrset == 'w':    hrsets  = ['wesaturate']
    if hrset == 'f':    hrsets  = ['flickr2k']
    if hrset == 'all':  hrsets  = ['diffract','wesaturate','flickr2k']

    hr_lists   = []
    train_imdb = []
    for hrset in hrsets:
        hr_folder = os.path.join('.//realsr_database',hrset,'HR')
        hr_lists  += glob.glob(os.path.join(hr_folder,'*.mat')) 
        print('reading synthetic training data from %s dataset with # %d imgs'%(hrset,len(hr_lists)))

    saver = tf.train.Saver(tf.trainable_variables()) 
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess,  ckpt_name.split('.meta')[0])
        s = time.time()
        for i_file in range(len(hr_lists)):
            hr_name = hr_lists[i_file]
            hr_img  = read_mat_y( hr_name )
            h,w = hr_img.shape
            hr_img  = np.pad( hr_img, ((0,h%2),(0,w%2)),'reflect')
            name = hr_name[-8:]

            if tag == 'bicubic':
                h,w = hr_img.shape
                lr_img = cv2.resize(hr_img,(w//int(factor),h//int(factor)),interpolation=cv2.INTER_CUBIC)
                lr_img = cv2.resize(lr_img,(w,h),interpolation=cv2.INTER_CUBIC)
            elif 'md' in tag:
                lr_img = hr_img
            else :
                lr_img = sess.run(train_output,feed_dict={train_input:hr_img[np.newaxis,:,:,np.newaxis],prob:1.0})
                lr_img = np.squeeze(np.array(lr_img))

            if mode == 'v3':
                lr_img = cv2.resize(lr_img,(hr_img.shape[1],hr_img.shape[0]))
            hr_img = hr_img[border:-border,border:-border]
            lr_img = lr_img[border:-border,border:-border]
            train_imdb.append([hr_img,lr_img])
        t = time.time()
        print('----------time for synthetic LR images: %ds---------------' % (t-s))
    return train_imdb
def get_syn_test_lists(dataset):
    # dataset in ['BSD68','Set5','Set14']
    path = os.path.join('.','realsr_database',dataset)
    test_lists = glob.glob(os.path.join(path,'*.*'))
    return test_lists

def get_div2k_syn_data(factor,tag,border,hrset='all',mode='v2'):

    if hrset == 'd':    hrsets  = ['diffract']
    if hrset == 'w':    hrsets  = ['wesaturate']
    if hrset == 'f':    hrsets  = ['flickr2k']
    if hrset == 'all':  hrsets  = ['diffract','wesaturate','flickr2k']

    hr_lists   = []
    train_imdb = []
    for hrset in hrsets:
        hr_folder = os.path.join('.//realsr_database',hrset,'HR')
        hr_lists  += glob.glob(os.path.join(hr_folder,'*.mat')) 
        print('reading synthetic training data from %s dataset with # %d imgs'%(hrset,len(hr_lists)))

    for i_file in range(len(hr_lists)):
        hr_name = hr_lists[i_file]
        if mode == 'v2':
            hr_img = scipy.io.loadmat( hr_name )['img']
            h,w = hr_img.shape[:2]
            hr_img  = np.pad( hr_img, ((0,h%2),(0,w%2)),'reflect')
        else:
            hr_img  = read_img( hr_name )
            h,w = hr_img.shape[:2]
            hr_img  = np.pad( hr_img, ((0,h%2),(0,w%2),(0,0)),'reflect')
        name = hr_name[-8:]

        if tag == 'bicubic':
            h,w = hr_img.shape
            lr_img = cv2.resize(hr_img,(w//int(factor),h//int(factor)),interpolation=cv2.INTER_CUBIC)
            lr_img = cv2.resize(lr_img,(w,h),interpolation=cv2.INTER_CUBIC)
        elif 'md' in tag:
            lr_img = hr_img
        elif 'kmsr' in tag:
            lr_img = hr_img
        else :
            lr_name = os.path.join('.//realsr_database',hrset,'synthetic',factor,name.replace('.mat','_'+tag + '.png'))
            lr_img = read_img(lr_name )

        # if mode == 'v3':
        #     lr_img = cv2.resize(lr_img,(hr_img.shape[1],hr_img.shape[0]))
        hr_img = hr_img[border:-border,border:-border]
        lr_img = lr_img[border:-border,border:-border]
        train_imdb.append([hr_img,lr_img])
    return train_imdb


def get_sr_lists(factor,sze,stride,mode='v2'):
    if mode =='v2':
        folder_name = './/realsr_database//RealSR_ycc'
    else:
        folder_name = './/realsr_database//RealSR_v3'
    folder = [os.path.join(folder_name,'Canon//Train'),os.path.join(folder_name,'Nikon//Train')]
    
    folder = [os.path.join(i,factor) for i in folder]
    train_lists = []
    hr_lists = glob.glob(os.path.join(folder[0],'*HR.png')) 
    hr_lists += glob.glob(os.path.join(folder[1],'*HR.png'))
    train_lists = [[i,i.replace('HR','LR'+factor)] for i in hr_lists]

    lr_patches = []
    hr_patches = []
    train_imdb = []
    print('reading and cropping training data from real sr dataset #imgs: %d'%len(train_lists))
    for i_samp in range(len(train_lists)):
        
        hr_name,lr_name = train_lists[i_samp]
        hr_img = read_img( hr_name )# read_mat_y(hr_name)
        lr_img = read_img( lr_name )
        if mode != 'shuffle':
            lr_img = cv2.resize(lr_img,(hr_img.shape[1],hr_img.shape[0]))
        # lr_patches += crop_train_set(lr_img,sze,stride)
        # hr_patches += crop_train_set(hr_img,sze,stride)
        train_imdb.append( [ hr_img,lr_img ])
    # train_imdb = [ [hr_patches[i],lr_patches[i]] for i in range(len(hr_patches))]
    
    return train_imdb, train_lists

def kernel_vis(sess,kernel_basis,save_folder,epoch,N,sze):
    k_basis = np.squeeze(np.array( sess.run(kernel_basis)))
    k_basis = np.reshape(k_basis,(sze,-1),order='F')
    temp = [i/(k_basis.shape[1]-1)*k_basis.max() for i in range(0,k_basis.shape[1])]
    k_basis = np.concatenate((k_basis,np.reshape(temp,(1,-1))),axis=0)
    plt.imshow(k_basis)
    plt.savefig(os.path.join(save_folder,'basis_epo_%03d.png'%(epoch)))
    plt.close()
    return 0


def get_sr_patches_lists(factor,sze,stride,mode='v2',scale=2):
    if mode =='v2':
        folder_name = './/realsr_database//RealSR_ycc'
    elif mode =='v3':
        folder_name = './/realsr_database//RealSR_v3'
    folder = [os.path.join(folder_name,'Canon//Train'),os.path.join(folder_name,'Nikon//Train')]
    folder = [os.path.join(i,factor) for i in folder]
    train_lists = []
    hr_lists = glob.glob(os.path.join(folder[0],'*HR.png')) 
    hr_lists += glob.glob(os.path.join(folder[1],'*HR.png'))
    train_lists = [[i,i.replace('HR','LR'+factor)] for i in hr_lists]

    lr_patches = []
    hr_patches = []
    train_imdb = []
    print('reading and cropping training data #imgs: %d'%len(train_lists))
    print('from folder: %s'%folder_name)


    # train_lists = train_lists[1:20]
    for i_samp in range(len(train_lists)):
        hr_name = train_lists[i_samp][0]
        lr_name = train_lists[i_samp][1]
        hr_img = read_img( hr_name )
        lr_img = read_img( lr_name )

        # lr_img = read_dns_mat_y( lr_name )
        if mode == 'v3':
            lr_img = cv2.resize(lr_img,(hr_img.shape[1],hr_img.shape[0]))
            lr_patches += im2patches(lr_img,sze,stride)
            hr_patches += im2patches(hr_img,sze,stride)
            # lr_p, hr_p = crop_v3_patches(lr_img,hr_img,sze,stride,int(scale))
            # lr_patches += lr_p
            # hr_patches += hr_p
        else:
            lr_patches += im2patches(lr_img,sze,stride)
            hr_patches += im2patches(hr_img,sze,stride)
        # train_imdb.append( [ hr_img,lr_img ])
    train_imdb = [ [hr_patches[i],lr_patches[i]] for i in range(len(hr_patches))]
    train_lists = [[i,i.replace('HR','LR'+factor)] for i in hr_lists]

    return train_imdb, train_lists


def obtain_tag(train_list,offset,BATCH_SIZE):
    tag = []
    for i_file in range(BATCH_SIZE):
        tag.append( train_list[i_file][2] )
    tag = np.array( tag )
    tag = tag[:,np.newaxis,np.newaxis,np.newaxis]
    return tag


def crop_v3_patches(lr_img,hr_img,sze,stride,scale):
    sze = sze //scale
    stride = stride // scale
    h,w = lr_img.shape[:2]
    h_range = list(range(0,h-sze,stride))
    h_range.append(h-sze)
    w_range = list(range(0,w-sze,stride))
    w_range.append(w-sze)

    lr_patches = []
    hr_patches = []
    for x in h_range:
        for y in w_range:
            hr_patches.append(hr_img[scale*x:scale*x+scale*sze,scale*y:scale*y+scale*sze])
            lr_patches.append(lr_img[x:x+sze,y:y+sze])
    return lr_patches,hr_patches


def apply_kpn_basis_color(train_input,coefs,kernel_basis,sze,N,ch=3):
    train_input_r,train_input_g,train_input_b = tf.split( train_input,ch,axis=-1)
    coefs_r,coefs_g,coefs_b = tf.split( coefs, ch, axis=-1)
    
    train_output_r = apply_kpn_basis( train_input_r,kernel_basis,coefs_r)
    train_output_g = apply_kpn_basis( train_input_g,kernel_basis,coefs_g)
    train_output_b = apply_kpn_basis( train_input_b,kernel_basis,coefs_b)
    train_output = tf.concat((train_output_r,train_output_g,train_output_b),axis=-1)
    return train_output

def apply_kpn_basis(inp,k_basis,kernel_size,coefs):
    pad_sze = (kernel_size-1)//2
    inp = tf.pad(inp,[[0,0],[pad_sze,pad_sze],[pad_sze,pad_sze],[0,0]],'SYMMETRIC')
    # response = tf.nn.depthwise_conv2d(inp,k_basis, [1, 1, 1, 1], padding='VALID')
    response = tf.nn.conv2d(inp,k_basis,[1,1,1,1], padding='VALID')
    # out = tf.nn.depthwise_conv2d(inp,k_basis, [1, 1, 1, 1], padding='VALID')
    out = tf.multiply(response, coefs)
    out = tf.reduce_sum( out,axis=-1,keepdims=True)
    return out,response

def apply_deep_kpn_basis(inp,basis_group,ks_group,kernel_size,coefs):
    layers = len(basis_group)
    
    pad_sze = (kernel_size-1)//2
    inp = tf.pad(inp,[[0,0],[pad_sze,pad_sze],[pad_sze,pad_sze],[0,0]],'SYMMETRIC')
    
    response = tf.nn.conv2d(inp,basis_group[0],[1,1,1,1], padding='VALID')
    for i_layer in range(layers-1):
        response = tf.nn.depthwise_conv2d(response,tf.transpose(basis_group[i_layer+1],(0,1,3,2)), 
                        [1, 1, 1, 1], padding='VALID')

    out = tf.multiply(response, coefs)
    out = tf.reduce_sum( out,axis=-1,keepdims=True)
    return out,response

def build_degrade_net(x,num_k_basis,out_ch,reuse = False,scope = 'degrade'):
    # stack consecutive convs
    # x = stack_convs(x,ch=32,layers=9,reuse=reuse)
    # use unet to extract features
    x = extract_feature_unet(x, reuse,scope,layers=10,ch=32)

    x = conv_layer( x,ch=num_k_basis*out_ch,filter_size=7,use_bias = True,name='conv_final',reuse=reuse)
    x = tf.nn.sigmoid(x)
    # x = tf.nn.dropout(x,self.prob)
    # x = tf.nn.softmax(x,-1)
    return x

def deep_basis_generator(inp,basis_group,ks_group,kernel_size,coefs):
    layers = len(basis_group)
    
    pad_sze = (kernel_size-1)//2
    inp = tf.pad(inp,[[0,0],[pad_sze,pad_sze],[pad_sze,pad_sze],[0,0]],'CONSTANT')
    
    response = tf.nn.conv2d(inp,basis_group[0],[1,1,1,1], padding='SAME')
    for i_layer in range(layers-1):
        response = tf.nn.depthwise_conv2d(response,tf.transpose(basis_group[i_layer+1],(0,1,3,2)), 
                        [1, 1, 1, 1], padding='SAME')
    response = tf.transpose(response,[1,2,0,3])
    return response


def degrade_kpn(x, reuse,scope,layers=10,ch=32):
    inp = x 
    x = extract_feature_unet(x, reuse,scope,layers=10,ch=32)
    kernels = conv_layer( x, ch=15*15,filter_size=7,use_bias=True,name='conv_final',reuse=reuse)
    train_output = [kpn_module(tf.expand_dims(inp[:,:,:,c],-1),kernels,ks=15,bs=16,channels=1) for c in range(3)]
    train_output = tf.concat( train_output, -1)
    return train_output,kernels

def kpn_module(input_tensor,per_pixel_weight,ks=3,bs=64,channels=3):
    bs = tf.shape(input_tensor)[0]
    h = tf.shape(input_tensor)[1]
    w = tf.shape(input_tensor)[2]
    pad_num = int( (ks-1)/2 )
    x = tf.pad( input_tensor,[[0,0],[pad_num,pad_num],[pad_num,pad_num],[0,0]], mode='symmetric')
    input_matrix = tf.image.extract_image_patches(x, [1,ks,ks,1],[1,1,1,1],[1,1,1,1],"VALID")

    kernel = tf.reshape( per_pixel_weight,(bs,h,w,ks*ks,channels)) # kernel of size:bs*hw*ks^2*ch
    input_matrix  = tf.reshape( input_matrix, (bs,h,w,ks*ks,channels))
    result = tf.multiply(kernel, input_matrix) # size bs*(hw)*(ks*ks)*chanel
    result = tf.reduce_sum( result, -2) # size bs*(hw)*1*channels
    # tensor = tf.reshape( result, (bs, height, width,channels))
    return result

def sum_to_one_loss(kernel,N ):
    kernel = tf.reduce_sum( kernel,[0,1])
    loss = tf.reduce_mean( tf.abs(kernel - 1))
    return loss

def boundry_loss(kernel,sze,N):
    w = np.zeros((sze,sze))
    c = (sze -1 )//2
    sigma = 3
    for x in range(0,sze):
        for y in range(0,sze):
            w[x,y] = np.exp( ((x-c)**2 + (y-c)**2/sigma))
    
    w = np.tile( w[:,:,np.newaxis,np.newaxis],(1,1,1,N))
    loss = tf.reduce_sum( tf.abs( tf.multiply( kernel,w)))
    return loss

def centroid_loss(kernel,sze,N):
    c = (sze - 1)//2
    indice = np.array( range(0,sze))
    indice = (indice[:,np.newaxis,np.newaxis] - c).astype('float32')
    row_sum = tf.reduce_sum( kernel,0)
    col_sum = tf.reduce_sum( kernel,1)
    k_sum   = tf.reduce_sum( kernel,[0,1])
    r_loss = tf.reduce_sum(tf.multiply( tf.tile(indice,(1,1,N)),row_sum),axis=0)/k_sum
    loss = tf.reduce_mean(tf.abs( r_loss ))
    c_loss = tf.reduce_sum(tf.multiply( tf.tile(indice,(1,1,N)),col_sum),axis=0)/k_sum
    loss += tf.reduce_mean(tf.abs( c_loss ))
    
    # loss  = tf.reduce_sum(tf.multiply( x,kernel)) + tf.reduce_sum(tf.multiply( y,kernel))
    return loss


def get_tv_loss( im ):
    # x of shape bs*H*W*channels, calculate the tv per channel
    grad_x = im[:,:,:-1,:] - im[:,:,1:,:]
    grad_y = im[:,:-1,:,:] - im[:,1:,:,:]
    loss = tf.reduce_sum( tf.abs( grad_x )) + tf.reduce_sum(tf.abs( grad_y ))
    return loss

def read_dns_mat_y(filename):
    im = scipy.io.loadmat(filename)['dns_img'].astype('float32')
    return im

def gause_dns( im , sigma=0.4):
    out = []
    g_filter = gkernel(9, sigma )
    for i_img in range(im.shape[0]):
        temp = signal.convolve2d(im[i_img,:,:,:].squeeze(),g_filter,boundary='symm',mode='same')
        out.append(temp)
    out = np.array( out )
    return out[:,:,:,np.newaxis] 

def read_mat_y(filename):
    im = scipy.io.loadmat(filename)['img']
    return im


def load_degrade_imdb( imdb_path, camera,tag ):
    folder = os.path.join( './/database', camera , 'tif_reg//train')
    imdb = []
    labels = sorted( glob.glob(os.path.join(folder, '*_7.1_reg.png')))

    sze = 128
    stride =  90

    for i_fold in range(len(labels)):
        group_name = labels[i_fold].split('_7.1')[0][-6:]
        print( 'processing image No.',group_name)
        # group_name = '000014'
        label_name = os.path.join( folder, group_name + '_7.1_reg.png')
        label  = crop_patch( label_name,sze,stride,border=3)

        border = 3
        # when using learned kernel to  apply 
        f22_name = os.path.join( folder, group_name + '_'+ tag +'.png')
        im_f22 = crop_patch( f22_name ,sze,stride,border=3)
        '''
        # testing, generating synthetic degrade imgs
        im = scipy.misc.imread(label_name).astype('float32')/255
        h,w,c=im.shape
        kernels = gkernel(size = 7, sigma =1.0) 
        r = signal.convolve2d(im[:,:,0],kernels,boundary='symm',mode='same')
        g = signal.convolve2d(im[:,:,1],kernels,boundary='symm',mode='same')
        b = signal.convolve2d(im[:,:,2],kernels,boundary='symm',mode='same')
        t = np.concatenate((r[:,:,np.newaxis],g[:,:,np.newaxis],b[:,:,np.newaxis]),axis=-1)
        t = t[border:-border,border:-border,:]
        im_f22 = im_to_patch( t, sze, stride)
        '''
        f18_name = os.path.join( folder, group_name + '_18.0_reg_c.png')
        im_f18 = crop_patch( f18_name ,sze,stride,border=3)

        f14_name = os.path.join( folder, group_name + '_14.0_reg_c.png')
        im_f14 = crop_patch( f14_name ,sze,stride,border=3)


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


def g_kernel_init( sze,N ):
    np.random.seed(0)
    k_init = []
    sigma_x = [[2,2],[2,3],[1,3],[3,4]]
    for i in range(np.max([N//4,1])):
        t = anistropic_gaussian_v2(sze, sigma_x[i][0], sigma_x[i][1])
        k_init.append( t )
        k_init.append(np.transpose( t))
        k_init.append(rotate(t,45,reshape=False))
        k_init.append(rotate(t,-45,reshape=False))
    k_init = np.array( k_init )
    # k_init += (np.random.rand( N,sze,sze )-0.5)*0.03 # add noise to init
    # t  = np.reshape(k_init,(-1,15))
    # plt.imshow(t );plt.show()
    k_init = np.expand_dims(np.transpose(k_init,[1,2,0]),2)
    k_init = k_init / (np.sum( k_init,(0,1),keepdims=True)+1e-6) /(N//4)
    init = tf.constant_initializer(k_init )
    return init


def get_MDKernel(factor ):
    data = scipy.io.loadmat('.//SRMDX'+factor+'Kernel.mat')
    AtrGK = data['AtripGK'] 
    return AtrGK

def multiple_degradation( hr_tensor,factor,AtrGK):
    h = hr_tensor.shape[1]
    w = hr_tensor.shape[2]
    lr_tensor = []
    for bs in range(hr_tensor.shape[0]):
        t = np.random.randint(low=0,high=(AtrGK.shape[-1]))
        kernel = AtrGK[:,:,0,t].squeeze()
        lr_img = signal.convolve2d(hr_tensor[bs,:,:,:].squeeze(),kernel,boundary='symm',mode='same')
        
        # lr_img = cv2.resize(lr_img,(w//int(factor),h//int(factor)),interpolation=cv2.INTER_CUBIC)
        # lr_img = cv2.resize(lr_img,(w,h),interpolation=cv2.INTER_CUBIC)
        lr_tensor.append(lr_img)
    lr_tensor = np.array(lr_tensor)
    lr_tensor = lr_tensor[:,:,:,np.newaxis]
    return lr_tensor

def random_init( sze, N):
    np.random.seed(0)
    init = np.random.rand(sze,sze,1,N)
    init = init.astype('float32')
    if N > 1:
        init = init / (np.sum( init,(0,1),keepdims=True)+1e-6) /(N/4)
    else:
        init = init / (np.sum( init,(0,1),keepdims=True)+1e-6)
    init = tf.constant_initializer( init )
    return init

def coefs_vis(th_coefs,sze,path,epoch,N,border):
    th_coefs = th_coefs[0,border:-border,border:-border,:].squeeze()
    # th_coefs = np.reshape(np.transpose(th_coefs,(0,2,1)),(sze,-1))
    for i in range(N//4):
        for j in range(4):
            plt.subplot(N//4,4,i*4+j+1)
            plt.imshow(th_coefs[:,:,i*4+j])
    plt.savefig(os.path.join( path,'vis','coefs_epo_%03d.png'%(epoch)))
    plt.close()
    return 0

def build_shuffle_model(train_input,scale,channels,reuse,scope):

    ch = 64
    ## plain CNN architecture
    # x = stack_convs(train_input,64,19,reuse,name='conv',mode='conv_first')
    # x = conv_layer(x,scale**2*channels,3,True,'final_conv',reuse,paddingMode='same')
    # train_output = shuffle_up(x,channels,scale)
    ## ResNet architecture
    x = conv_layer(train_input,ch,3,True,'conv_first',reuse,paddingMode='same')
    for i in range(8):
        x = res_block( x,ch,3,'ResBlock_%02d_'%(i),reuse)
    # x = conv_layer(x,scale**2*channels,3,True,'final_conv',reuse,paddingMode='same')
    # train_output = shuffle_up(x,channels,scale)
    train_output = conv_layer(x,channels,3,True,'final_conv',reuse,paddingMode='same')
    return train_output

def build_shuffle_mid_model(train_input,scale,channels,reuse,scope):

    ch = 64
    x = conv_layer(train_input,ch,3,True,'conv_first',reuse,paddingMode='same')
    for i in range(8):
        x = res_block( x,ch,3,'ResBlock_%02d_'%(i),reuse)
    ch = 32
    x = conv_layer(x,scale**2*ch,3,True,'shuffle_conv',reuse,paddingMode='same')
    x = shuffle_up(x,ch,scale)
    x = stack_convs(x,ch,2,reuse,name='recon_conv',mode='conv_first',fs=3)
    train_output = conv_layer(x,channels,3,True,'final_conv',reuse,paddingMode='same')
    return train_output

def get_kmsr_train_batch(D_kernels, train_list, offset,sze, BATCH_SIZE):
    inp = []
    label = []
    
    for i_file in range(BATCH_SIZE):
        kernel = random.choice(D_kernels)
        hr_im = train_list[i_file][0]
        H,W = hr_im.shape
        h = np.random.randint(0,H-sze-1)
        w = np.random.randint(0,W-sze-1)
        hr_patch = hr_im[h:h+sze,w:w+sze]
        lr_patch = ndimage.filters.convolve(hr_patch, kernel, mode='reflect')
        label.append( hr_patch)
        inp.append( lr_patch )
        
    inp = np.array(inp)
    inp = np.expand_dims(inp,axis=-1)
    label = np.array(label)
    label = np.expand_dims(label,axis=-1)

    return label,inp

def get_kmsr_kernels( factor ):
    D_kernels = []
    # gan generated kernels
    folder = './kmsr/' + factor
    gan_kernels = sorted(glob.glob(os.path.join(folder,'*.mat')))
    for kernel_name in gan_kernels:
        D_kernels.append( scipy.io.loadmat(kernel_name)['kernel'] )
    # optimization based kernels
    folder = './kmsr/x' + factor + 'results'
    folders = os.listdir( folder )
    for name in folders:
        kernel_name = os.path.join( folder, name , 'kernel.mat' )
        D_kernels.append( scipy.io.loadmat(kernel_name)['k'] )
    return D_kernels
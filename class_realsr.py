import os, glob, re, signal, sys, argparse, threading, time, h5py, math, random, logging
import scipy.misc 
import scipy.io
from skimage.measure import compare_ssim
from scipy import signal
from random import shuffle
from PIL import Image
from sklearn.feature_extraction import image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from utils_degrade import *
from build_rcan import build_rcan
from config import *
import lpips_tf
from imresize import imresize
class realsr(object):
    def __init__(self, args):
        self.args = args
        self.channels = 1
        self.BATCH_SIZE = args.BATCH_SIZE
        self.MAX_EPOCH  = args.MAX_EPOCH
        self.IMG_SIZE   = args.IMG_SIZE
        self.mode       = args.mode
        self.learning_rate = args.BASE_LR
        self.factor = args.factor
        self.sigma  = float(args.sigma)
        self.data   = args.data
        self.hrset  = args.hrset
        # test lists
        self.test_lists = get_sr_test_lists(self.factor)

    def test_vis_result(self,model_name,test_path,out_path,tag,mode='mat'):  
        test_sze = 400
        stride = 300
        if not out_path: 
            out_path = os.path.join('./test_vis','sr_x%s'%self.factor)
        self.model_path = os.path.join('./models','sr_x'+self.factor,'checkpoints_'+ model_name) 
        ###############################################
        self.mode = 'vdsr'
        self.model_path   = os.path.join('.//models//sr_x3','checkpoints_r_k15n8_vdsr_syn')
        save_name = 'dml_vdsr_syn'
        ###############################################
        model_ckpt = get_model_lists(self.model_path)[-3]

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
            self.test_input  = tf.placeholder(tf.float32, shape=(None, None, None, self.channels))
            self.prob        = tf.placeholder(tf.float32)
            if self.mode == 'vdsr':
                self.test_output = self.build_vdsr(x=self.test_input, reuse=self.reuse, scope='sr_unet',training = False)
            elif self.mode == 'rcan':
                self.test_output,_ = build_rcan(x=self.test_input, reuse=self.reuse,ch=1)
            self.saver = tf.train.Saver() 
            self.saver.restore(sess,  model_ckpt.split('.meta')[0])
            # for learn to zoom fold
            if mode == 'mat':
                test_lists = glob.glob(os.path.join(test_path,'*_'+tag+'.mat'))
            elif mode == 'rgb':
                test_lists = glob.glob(os.path.join(test_path,'*.png'))
                print(len(test_lists))
            for i_test in range(len(test_lists)):
                print('testing on Learn to zoom dataset, #img %d'%i_test)
                name = test_lists[i_test][-13:-8]
                if mode == 'mat':
                    test_img = read_mat_y( test_lists[i_test] )
                elif mode == 'rgb':
                    test_img_rgb = read_img( test_lists[i_test] ) 
                    # test_img_ycc = read_mat_y( test_lists[i_test])
                    test_ycc = cv2.cvtColor(test_img_rgb, cv2.COLOR_RGB2YCR_CB)
                    test_img = test_ycc[:,:,0].squeeze()
                    # apply gaussion smoothing to remove jpg compression artifacts
                # if rgb
                output = test_patch(test_img, self.test_input, self.test_output,sess,self.prob)
                
                if mode == 'rgb':
                    test_ycc[:,:,0] = output
                    output = cv2.cvtColor(test_ycc, cv2.COLOR_YCR_CB2RGB)
                    save_png(os.path.join(out_path,(name+'_'+save_name+'.png')),output)
                else:
                    save_png(os.path.join(out_path,name.replace('.mat','_'+save_name+'.png')),output)
        return 0

    def test(self,model_name,out_path=None):  
        if not out_path: 
            out_path = os.path.join('./test_vis','degradation','sr_x%s'%self.factor)
        self.model_path = os.path.join('./models','sr_x'+self.factor,'checkpoints_'+ model_name + self.mode+'_'+self.data) 
        self.model_path = os.path.join('./models','sr_x'+self.factor,'checkpoints_r_k15n8_rcan_syn_real_lpips' )

        save_tag = '_dml_vdsr.png'
        save  = False
        ##########
        model_list = get_model_lists(self.model_path)
        _,_,model_list = check_checkpoints(self.model_path)

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
            self.test_input  = tf.placeholder(tf.float32, shape=(None, None, None, self.channels))
            self.test_gt  = tf.placeholder(tf.float32, shape=(None, None, None, self.channels))
            lpips_index   = lpips_tf.lpips(tf.tile(self.test_input,(1,1,1,3)), tf.tile(self.test_gt,(1,1,1,3)),
                            model='net-lin', net='alex')
            self.prob        = tf.placeholder(tf.float32)
            if self.mode == 'vdsr':
                self.test_output = self.build_vdsr(x=self.test_input, reuse=self.reuse, scope='sr_unet',training = False)
            elif self.mode == 'rcan':
                self.test_output,_ = build_rcan(x=self.test_input, reuse=self.reuse,ch=1)
            self.saver = tf.train.Saver()  
            
            psnr = np.zeros([len(model_list),len(self.test_lists)])
            ssim = np.zeros([len(model_list),len(self.test_lists)])
            lpips = np.zeros([len(model_list),len(self.test_lists)])
            best_psnr = 0
            best_lpips = 1
            
            
            for i_model in range(len(model_list)):
                model_ckpt = model_list[-1-i_model]
                # model_ckpt = model_list[-23]
                epoch = int(model_ckpt.split('epoch_')[-1].split('.')[0])
                print( "real sr Testing state for epoch %03d"%(epoch))
                
                self.saver.restore(sess, model_ckpt.split('.meta')[0])
                for i_group in range(len(self.test_lists)):
                    hr_name,lr_name = self.test_lists[i_group]
                    if '.mat' in hr_name:
                        hr_img = read_mat_y( hr_name )
                        lr_img = read_mat_y( lr_name )
                    elif '.png' in hr_name:
                        hr_img_rgb = read_img( hr_name)
                        lr_img_rgb = read_img( lr_name)
                        hr_img, lr_img = [hr_img_rgb, lr_img_rgb]
                        # hr_img_ycc = rgb2ycbcr_matlab( hr_img_rgb)
                        # lr_img_ycc = rgb2ycbcr_matlab( lr_img_rgb)
                        # hr_img     = hr_img_ycc[:,:,0]
                        # lr_img     = lr_img_ycc[:,:,0]
                    name = lr_name[-17:]

                    start_t = time.time()
                    if self.mode == 'rcan':
                        output = test_patch(lr_img, self.test_input, self.test_output,sess,self.prob)
                    elif self.mode == 'vdsr':
                        output = sess.run(self.test_output,feed_dict={self.test_input: lr_img[np.newaxis,:,:,np.newaxis],
                                                        self.prob:1.0 })
                    output = np.squeeze( np.array(output) )
                    end_t   = time.time()
                    psnr[i_model,i_group] = cal_psnr(output,hr_img)
                    lpips[i_model,i_group] = sess.run(lpips_index, feed_dict={self.test_input:output[np.newaxis,:,:,np.newaxis], 
                                            self.test_gt: hr_img[np.newaxis,:,:,np.newaxis]})
                    # ssim[i_model,i_group] = compare_ssim(output,hr_img,multichannel=False)
                    ssim[i_model,i_group] = 0
                    print( "sr testing img:%02d,  time %.2f psnr: %.2f\t"%(i_group,end_t-start_t,psnr[i_model,i_group]))
                    if save:
                        output_rgb = ycbcr2rgb_matlab( np.transpose( np.array( 
                                    [output,lr_img_ycc[:,:,1],lr_img_ycc[:,:,2]]),(1,2,0)) )
                        save_png( os.path.join(out_path,name.replace('.png',save_tag)), output_rgb)

                if psnr[i_model,:].mean() > best_psnr:
                    best_psnr  = psnr[i_model,:].mean()
                    best_ckpt  = model_ckpt

                if lpips[i_model,:].mean() <best_lpips:
                    best_lpips  = lpips[i_model,:].mean()
                print( "this psnr %.4f\t best psnr %.4f\t "%(psnr[i_model,:].mean(),best_psnr ))
                print( "this lpips %.4f\t best lpips %.4f\t "%(lpips[i_model,:].mean(),best_lpips ))
                print( "this ssim %.4f\t "%(ssim[i_model,:].mean() ))
                print( self.model_path ) 
                print(np.mean(lpips,-1))
            print( self.model_path )        
            print( "best psnr %.4f\t "%(psnr.mean(1).max()))

        return psnr


    def degradation(self):
        self.degrade_model_path = os.path.join('.//sr_degrade', 'checkpoints_' + 'x%s_'%self.factor + self.degrade_model_name ) 
        make_dir(self.degrade_model_path)
        make_dir(os.path.join(self.degrade_model_path,'vis'))

        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        train_input = tf.placeholder(tf.float32, shape=(None,None,None, self.channels))
        train_label = tf.placeholder(tf.float32, shape=(None,None,None, self.channels))
        self.prob   = tf.placeholder(tf.float32)
        coefs = build_degrade_net(train_input,self.num_k_basis,self.channels,reuse = False,scope = 'degrade') 

        
        if 'g_' in self.degrade_model_name :
            init = g_kernel_init( self.k_size,self.num_k_basis )
            kernel_basis = tf.get_variable('k_basis',[self.k_size,self.k_size,1,self.num_k_basis],initializer=init)
        elif 'r_' in self.degrade_model_name :
            init = random_init(self.k_size, self.num_k_basis)
            kernel_basis = tf.get_variable('k_basis',[self.k_size,self.k_size,1,self.num_k_basis],initializer=init) # initializer=init
        elif 'deep_' in self.degrade_model_name :
            self.k_group  = [tf.get_variable('k%02d'%i,[self.ks_group[i],self.ks_group[i],1,self.num_k_basis],
                            initializer=tf.random_normal_initializer) for i in range(len(self.ks_group))]
            kernel_basis  = deep_basis_generator(tf.ones([1,1,1,1]),self.k_group,self.ks_group,self.k_size,coefs)
            
        # kernel_basis   = tf.nn.sigmoid(kernel_basis_o)
        ## kernel regularizer
        # b_loss = boundry_loss(kernel_basis,self.k_size,self.num_k_basis)
        sum_loss = sum_to_one_loss(kernel_basis,self.num_k_basis)
        # c_loss = centroid_loss(kernel_basis,self.k_size,self.num_k_basis )
        # kernel basis network
        
        if 'deep_' in self.degrade_model_name :
            train_output,self.response = apply_deep_kpn_basis(train_input,self.k_group,self.ks_group,self.k_size,coefs)
        else:
            train_output,self.response = apply_kpn_basis(train_input,kernel_basis,self.k_size,coefs)
         
        border = (self.k_size-1)//2
        residual = train_output[:,border:-border,border:-border,:] - train_label[:,border:-border,border:-border,:] 
        # residual = train_output - train_label
        mse_loss = tf.reduce_sum( tf.nn.l2_loss( residual ))
        tv_loss =  get_tv_loss( coefs)  # add smooth prior on coefs
        tv_img  =  get_tv_loss( train_output[:,border:-border,border:-border,:] )
        loss = 1*mse_loss + 0*tv_loss +  sum_loss *0 + tv_img*0.0

        all_vars = tf.trainable_variables()
        # loss += tf.nn.l2_loss(all_vars[0])*1e-5

        lr = tf.train.piecewise_constant(global_step, boundaries=[6000],values=[1e-04,1e-04])
        optimizer = tf.train.AdamOptimizer(learning_rate=lr) 
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr*0.001) 
        opt   = optimizer.minimize(loss,  global_step=global_step, var_list= all_vars)
        # gvs = optimizer.compute_gradients(loss)
        # gvs = [(tf.clip_by_value(grad, -47000., 47000.), var) for grad, var in gvs]
        # opt = optimizer.apply_gradients(gvs,global_step=global_step)

        self.saver = tf.train.Saver(all_vars,global_step, max_to_keep=0)
        # training
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            tf.initialize_all_variables().run()
            
            ckpt_name,start,model_lists = check_checkpoints(self.degrade_model_path)
            if ckpt_name:
                self.saver.restore(sess,ckpt_name)
            # for evaluation only
            # test_psnr = []
            # test_ssim = []
            # for epoch, ckpt_name in enumerate(model_lists):
            #     self.saver.restore(sess,ckpt_name[:-5])
            #     th_psnr,th_ssim = self.evaluate_degrade_kernel(sess,100,'test',kernel_basis,train_output,train_input,save=False)
            #     test_psnr.append(th_psnr)
            #     test_ssim.append(th_ssim)
            #     print('results for epoch %02d'%(epoch))
            #     print(test_psnr)
            #     print(test_ssim)
            #     print('best psnr: %.2f'%(np.array(test_psnr).max()))
            #     print('best ssim: %.4f'%(np.array(test_ssim).max()))

            self.train_list,self.train_files = get_sr_patches_lists(self.factor,
                                        sze=self.IMG_SIZE[0],stride = self.IMG_SIZE[0]) 

            train_loss = np.zeros(self.degrade_EPOCH)
            test_psnr = []
            test_ssim = []
            kernel_vis(sess,kernel_basis,os.path.join(self.degrade_model_path,'vis'),-1,self.num_k_basis,self.k_size)
            for epoch in  range(start, self.degrade_EPOCH):
                error = 0
                shuffle(self.train_list)
                for step in range(len(self.train_list)//self.BATCH_SIZE):
                    offset = step*self.BATCH_SIZE
                    
                    hr_data,lr_data = self.get_image_batch( self.train_list,  offset, self.BATCH_SIZE)
                    hr_data,lr_data = data_augment( [hr_data,lr_data ])
                    feed_dict = {train_input:hr_data, train_label:lr_data,self.prob:0.5} # feed_dict: dict
                    s = time.time()
                    th_lr,_,l,l2,l_tv,k_sum,l_img_tv,output, g_step = sess.run([lr,opt, loss, mse_loss,tv_loss,
                                            sum_loss,tv_img,train_output, global_step],feed_dict=feed_dict)
                    t = time.time()
                    if (g_step-1) % 50 == 0:
                        print( "[epoch %d: %d/%d] mse loss %.4f,  psnr %.2f,  tv_loss %.4f, img tv loss %.4f time %.4f, "%( epoch,step, 
                                            len(self.train_list)//self.BATCH_SIZE,np.sum(l2)/self.BATCH_SIZE,
                                            20*math.log10(math.sqrt(self.IMG_SIZE[0]**2*self.channels/np.sum(l2)*self.BATCH_SIZE)),
                                            np.sum(l_tv)/(self.BATCH_SIZE*self.num_k_basis*self.channels*self.IMG_SIZE[0]**2),
                                            l_img_tv,(t-s)))
                    error = error + l2
                self.saver.save(sess, self.degrade_model_path+"/conv_1l_epoch_%03d.ckpt" % epoch)
                # evaluate each epoch, save basis kernels and basis coefs
                make_dir(os.path.join( self.degrade_model_path,'vis'))
                th_coefs = sess.run(coefs,feed_dict=feed_dict)
                coefs_vis(th_coefs,self.IMG_SIZE[0],self.degrade_model_path,epoch,self.num_k_basis,border)
                th_test_psnr,_ = self.evaluate_degrade_kernel(sess,epoch,'test',kernel_basis,train_output,train_input,save=False)
                print('kernel generation stage PSNR: %.2f\t for epoch : %03d'%(th_test_psnr, epoch))
                test_psnr.append( th_test_psnr )
                
                train_loss[epoch] = error / len(self.train_list)
                print( "[epoch %d] l2 loss %.4f\t"%( epoch, train_loss[epoch] ))
                plt.plot(train_loss)
                plt.savefig(os.path.join( self.degrade_model_path,'train_loss.png'))
                plt.close()
                
            trainset_psnr,_ = self.evaluate_degrade_kernel(sess,self.degrade_EPOCH,'train',kernel_basis,train_output,train_input,save=True)
            print('training set degrade model psnr: ')
            print(trainset_psnr)

            test_psnr = np.array( test_psnr )
            print('train loss:')
            print( train_loss )
            print( 'test psnr:')
            print(test_psnr )
            print( 'max test psnr:')
            print( test_psnr.max())
            print(self.degrade_model_path)
        return 0

    def evaluate_degrade_kernel(self,sess,epoch,tag,kernel_basis,train_output,train_input,save=False):
        # generating synthetic diffraction images
        save_folder = os.path.join( self.degrade_model_path,'vis')
        folder = ['.//realsr_database//RealSR_ycc//Canon//Train','.//realsr_database//RealSR_ycc//Nikon//Train']
        train_save_folder = os.path.join('.//realsr_database//RealSR_ycc//synthetic_lr',self.factor)
        if tag == 'train':
            validate_lists = self.train_files
        elif tag == 'test':
            validate_lists = self.test_lists
        self.tag = tag
        print('evaluation on %s set'%tag)
        # visualize learned degradation kernels
        kernel_vis(sess,kernel_basis,save_folder,epoch,self.num_k_basis,self.k_size)

        # generating images
        psnr = []
        border = (self.k_size-1)//2
        print('evaluation the synthetic HR psnr, #imgs:', len(validate_lists))

        for i_img in range( len( validate_lists)):
            hr_name, lr_name = validate_lists[i_img]
            if '.png' in hr_name:
                hr_img = read_img(hr_name)
                lr_img = read_img(lr_name)
            elif '.mat' in hr_name:
                hr_img  = read_mat_y( hr_name )
                lr_img  = read_mat_y( lr_name )
            h,w = hr_img.shape[:2]
            hr_img  = np.pad( hr_img, ((0,h%2),(0,w%2)),'reflect')


            syn_lr_img = sess.run(train_output,feed_dict={train_input:hr_img[np.newaxis,:,:,np.newaxis],self.prob:1.0})
            syn_lr_img = np.squeeze(np.array(syn_lr_img))
            syn_lr_img = syn_lr_img[:h,:w]
            psnr.append( cal_psnr(syn_lr_img[border:-border,border:-border],lr_img[border:-border,border:-border]))
            if False:
                name = lr_name[-17:]
                print('saving synthetic lr image on SR trainset for visualization, No. ',i_img)
                save_png_name,save_mat_name = [os.path.join(train_save_folder,
                                        name.replace('.mat','_'+self.degrade_model_name+tag))  for tag in ['.png','.mat']]
                save_png(save_png_name ,syn_lr_img)
                # scipy.io.savemat(save_name,{'img':syn_lr_img})

        if True:
            for hrset in ['diffract','wesaturate','flickr2k']:
                print('generating synthetic LR imgs on %s dataset...'%(hrset))
                folder = os.path.join('.//realsr_database',hrset,'HR')  # diffract
                make_dir(os.path.join('.//realsr_database',hrset,'synthetic'))
                save_folder = os.path.join('.//realsr_database',hrset,'synthetic',self.factor)
                make_dir(save_folder)
                imglists = glob.glob(os.path.join(folder,'*.mat'))
                print('saving training number: #%d...'%(len(imglists)))
                for i_img in range(len(imglists)):
                    hr_name = imglists[i_img]
                    name    = hr_name[-8:]
                    hr_img  = scipy.io.loadmat(hr_name)['img']
                    h,w     = hr_img.shape[:2]
                    hr_img  = np.pad( hr_img, ((0,h%2),(0,w%2)),'reflect')

                    syn_lr_img = sess.run(train_output,feed_dict={train_input:hr_img[np.newaxis,:,:,np.newaxis],self.prob:1.0})
                    syn_lr_img = syn_lr_img.squeeze()[:h,:w]
                    
                    save_png_name,save_mat_name = [ os.path.join(save_folder,
                                    name.replace('.mat','_'+self.degrade_model_name+tag)) for tag in ['.png','.mat']]
                    save_png(save_png_name ,syn_lr_img)
                    # scipy.io.savemat(save_mat_name,{'img':syn_lr_img})
        
        return np.array(psnr).mean(),0
    

    def train(self, model_name):
        # set model path  
        self.model_path = os.path.join('./models','sr_x'+self.factor,'checkpoints_'+ model_name + self.mode + '_' + self.data) 
        make_dir(self.model_path)
        


        print( 'training real sr model for scale: X%s'%(self.factor))
        print(self.model_path)
        # set net architecture
        self.global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        
        # set placeholder
        self.train_input = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.IMG_SIZE[0], self.IMG_SIZE[1], self.channels))
        self.train_gt  	 = tf.placeholder(tf.float32, shape=(self.BATCH_SIZE, self.IMG_SIZE[0], self.IMG_SIZE[1], self.channels))
        self.prob        = tf.placeholder(tf.float32)

        if 'kmsr' in model_name:    D_kernels = get_kmsr_kernels( self.factor )
        if 'md' in model_name:    AtrGK = get_MDKernel(self.factor )
        if self.data == 'real':
            self.train_list,self.train_files = get_sr_lists(self.factor,sze=self.IMG_SIZE[0],stride = self.IMG_SIZE[0]) 
        elif self.data == 'syn':
            self.train_list = get_div2k_syn_data(self.factor,self.tag, border=(self.k_size-1)//2,hrset = self.hrset)
        elif self.data == 'syn_real':
            self.train_list = get_div2k_syn_data(self.factor,self.tag,self.num_k_basis,self.k_size,self.ks_group,
                            border=(self.k_size-1)//2,hrset = self.hrset)
            for i in range(len(self.train_list)):    self.train_list[i].append(1)
            realsr_list,_ = get_sr_lists(self.factor,sze=self.IMG_SIZE[0],stride = self.IMG_SIZE[0]) 
            for i in range(len(realsr_list)):        realsr_list[i].append(0)
            self.train_list += realsr_list

        # for training on single-gpu
        if self.mode == 'rcan':
            self.train_output,_ = build_rcan(x=self.train_input,reuse=False,ch=1)
        elif self.mode == 'vdsr':
            self.train_output = self.build_vdsr(x=self.train_input, reuse=False, scope='sr_unet',training = True)
        self.perceptual_loss = tf.reduce_sum(lpips_tf.lpips(tf.tile(self.train_output,(1,1,1,3)),
                            tf.tile(self.train_gt,(1,1,1,3)), model='net-lin', net='alex'))
        
        self.loss =  tf.reduce_sum(tf.nn.l2_loss( self.train_output - self.train_gt ))
        # self.loss += 10*self.perceptual_loss
        


        # training vars
        self.all_vars = tf.trainable_variables()
        for var in self.all_vars:
            self.loss += tf.nn.l2_loss(var)*1e-5
        
        # setting learning rate and optimizer
        self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries=[1600*20],values=[0.0001,0.0001])
        optimizer  = tf.train.AdamOptimizer(learning_rate=self.learning_rate) 
        self.opt   = optimizer.minimize(self.loss,  global_step=self.global_step, var_list= self.all_vars)
        
        self.saver = tf.train.Saver(self.all_vars,self.global_step, max_to_keep=0)
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            file_writer = tf.summary.FileWriter(".//logs//"+model_name, sess.graph)
            tf.initialize_all_variables().run()
            
            # loading previous training models
            start = 0
            ckpt_name,start,model_lists = check_checkpoints(self.model_path)
            if ckpt_name:
                self.saver.restore(sess,ckpt_name)

            train_loss = np.zeros(self.MAX_EPOCH)
            valid_psnr = []
            valid_lpips = []
            best_psnr = 0
            for epoch in  range(start, self.MAX_EPOCH):
                error = 0
                shuffle(self.train_list)
                for step in range(len(self.train_list)//self.BATCH_SIZE):
                    offset = step*self.BATCH_SIZE
                    
                    if 'kmsr' in model_name:
                        hr_data,lr_data = get_kmsr_train_batch(D_kernels, self.train_list, offset, 
                                            self.IMG_SIZE[0], self.BATCH_SIZE)
                    else:
                        hr_data,lr_data = self.crop_image_patch(self.train_list, offset, self.BATCH_SIZE)
                    hr_data,lr_data = data_augment( [hr_data,lr_data ])
                    # adding noise to synthetic realistic LR image
                    if 'srmd' in model_name:
                        lr_data = multiple_degradation( hr_data ,self.factor,AtrGK )
                    if self.data == 'syn_real':
                        tag = obtain_tag(self.train_list,offset,self.BATCH_SIZE)
                        lr_data += tag * np.random.randn(self.BATCH_SIZE,self.IMG_SIZE[0],
                                                    self.IMG_SIZE[1],self.channels)*self.sigma/255.0
                    elif self.data == 'syn':
                        lr_data += np.random.randn(self.BATCH_SIZE,self.IMG_SIZE[0],
                                                    self.IMG_SIZE[1],self.channels)*self.sigma/255.0
                    feed_dict = {self.train_input:lr_data, self.train_gt:hr_data,self.prob:0.5} # feed_dict: dict
                    s = time.time()
                    lr,_,l2,th_lpips,output, g_step = sess.run([self.learning_rate,self.opt, self.loss, self.perceptual_loss,
                                        self.train_output, self.global_step],feed_dict=feed_dict)
                    t = time.time()
                    if (g_step-1) % 100 == 0:
                        print( "[epoch %d: %d/%d] l2 loss %.4f\t lpips: %.4f time %.4f\t"%( epoch,step, 
                                            len(self.train_list)//self.BATCH_SIZE,
                                            np.sum(l2)/self.BATCH_SIZE,np.sum(th_lpips)/self.BATCH_SIZE,(t-s)))
                    
                    error = error + l2
                    
                train_loss[epoch] = error / len(self.train_list)
                print( "[epoch %d] l2 loss %.4f\t"%( epoch, train_loss[epoch] ))
                validate_step = 10 if self.hrset in ['f','all'] else 50
                if (epoch+1 )% validate_step ==0:
                    th_valid_psnr,th_valid_lpips = self.validation(sess)
                    valid_psnr.append( th_valid_psnr )
                    valid_lpips.append( th_valid_lpips )
                    print( 'validate psnr, lpips on testing set: %.2f, %.4f'%(th_valid_psnr,th_valid_lpips))
                    if  th_valid_psnr >= np.array(valid_psnr).max():
                        self.saver.save(sess, self.model_path+"/SRunet_epoch_%03d.ckpt"%(epoch))
                    if  th_valid_lpips <= np.array(valid_lpips).min():
                        self.saver.save(sess, self.model_path+"/SRunet_epoch_%03d.ckpt"%(epoch))
                    print( 'validate metrics for epoches:')
                    print( np.array(valid_psnr))
                    print( 'best psnr, lpips on test dataset: %.2f, %.4f'%(np.array(valid_psnr).max(),
                                            np.array(valid_lpips).min()))

                    print(self.model_path)
            print(self.model_path)
                
        return 0

    
    def validation(self,sess):
        # generating synthetic diffraction images
        print('evaluation of the trained SR model ')
        test_input = tf.placeholder(tf.float32, shape=(None, None, None, self.channels))
        test_gt    = tf.placeholder(tf.float32, shape=(None, None, None, self.channels))
        lpips_index   = lpips_tf.lpips(tf.tile(test_input,(1,1,1,3)), tf.tile(test_gt,(1,1,1,3)),
                            model='net-lin', net='alex')
        if self.mode == 'rcan':
            test_output,_ = build_rcan(x=test_input, reuse=True,ch=1)
        elif self.mode == 'vdsr':
            test_output = self.build_vdsr(x=test_input, reuse=True, scope='sr_unet',training = False)
        
        psnr = []
        lpips = []
        for i_img in range( len( self.test_lists)):
            hr_name,lr_name = self.test_lists[i_img]
            hr_img = read_img( hr_name )
            lr_img = read_img( lr_name )
            if self.mode == 'vdsr':
                output = sess.run(test_output,feed_dict={test_input: lr_img[np.newaxis,:,:,np.newaxis],self.prob:1.0 })
            elif self.mode == 'rcan':
                output = test_patch(lr_img, test_input, test_output,sess,self.prob)

            output = np.array(output).squeeze()
            psnr.append( cal_psnr(output,hr_img))
            lpips.append( sess.run(lpips_index, feed_dict={test_input:output[np.newaxis,:,:,np.newaxis], 
                                        test_gt: hr_img[np.newaxis,:,:,np.newaxis]}))
        test_lpips_mean = np.array(lpips).mean()
        test_psnr_mean = np.array(psnr).mean()
        return test_psnr_mean,test_lpips_mean



    def build_sr_unet(self,x, reuse,scope,training):
        skip = x
        ch = 32
        global_state = []

        with tf.variable_scope(scope, reuse=reuse):
            x = conv_layer( x, ch,3,True,name='first_conv',reuse=reuse)
            x = tf.nn.relu(x)
            # encoding stage
            for i_level in range(2):
                for i_conv in range(2):
                    x = self.resblock(x,ch,'enc_l%02d_res%02d'%(i_level,i_conv),reuse )
                global_state.append( x )
                x = shuffle_down(x)
                ch = ch *4
            # decoding stage
            for i_level in range(2):
                for i_conv in range(2):
                    x = self.resblock(x,ch,'dec_l%02d_res%02d'%(i_level,i_conv),reuse )
                x = shuffle_up(x, ch//4 )
                x = tf.concat((x,global_state[-1-i_level]),axis=-1)
                ch = ch //4 
                x = conv_layer( x, ch,3,True,name='concat_conv%02d'%(i_level),reuse=reuse)
                x = tf.nn.relu(x )
            # final recovering stage
            x = self.resblock(x,ch,'rec_01',reuse )
            x = self.resblock(x,ch,'rec_02',reuse )
            x = tf.nn.dropout(x, self.prob )
            x = conv_layer( x, self.channels,filter_size=3,use_bias=True,name='final_conv',reuse=reuse)

        y = tf.add( x, skip)   
        return y


    def resblock(self,x,ch,name,reuse ):
        skip = x
        x = conv_layer(x,ch,3,True,name+'01',reuse )
        x = tf.nn.relu(x)
        x = conv_layer(x,ch,3,True,name+'02',reuse )
        x = x + skip 
        x = tf.nn.relu(x)
        return x 

    def get_image_batch(self, train_list,offset,batch_size):
        inp = []
        label = []
        for i_file in range(batch_size):
            inp.append( train_list[offset+i_file][1] )
            label.append(  train_list[offset+i_file][0] )
        inp = np.array(inp)
        inp = np.expand_dims(inp,axis=-1)
        label = np.array(label)
        label = np.expand_dims(label,axis=-1)
        return label,inp

    def crop_image_patch(self,train_list, offset, batch_size):
        inp = []
        label = []
        sze = self.IMG_SIZE[0]
        for i_file in range(batch_size):
            hr_im = train_list[i_file][0]
            lr_im = train_list[i_file][1]
            H,W = hr_im.shape
            h = np.random.randint(0,H-sze-1)
            w = np.random.randint(0,W-sze-1)
            label.append( hr_im[h:h+sze,w:w+sze])
            inp.append( lr_im[h:h+sze,w:w+sze])
        inp = np.array(inp)
        inp = np.expand_dims(inp,axis=-1)
        label = np.array(label)
        label = np.expand_dims(label,axis=-1)

        return label,inp

    def build_vdsr(self,x, reuse,scope,training):
        skip = x
        with tf.variable_scope(scope, reuse=reuse):
            x = stack_convs(x,ch=64,layers=19,reuse=reuse,name='conv',mode='conv_first')

            x = tf.nn.dropout(x, self.prob )
            x = conv_layer(x, ch=self.channels,filter_size=3,use_bias=True,name='conv_final',reuse=reuse)

        y = tf.add( x, skip)   
        return y

    def direct_degrade_net(self,x, reuse,scope,training,layers=10,ch=32):
        skip = x
        x = stack_resblock(x,32,layers,reuse,scope)
        x = tf.nn.relu(x )
        x = conv_layer( x, ch=self.channels,filter_size=7,use_bias=True,name='conv_final',reuse=reuse)
        y = tf.add( x, skip)   
        return y

    def direct_degrade_unet(self,x, reuse,scope,training,layers=10,ch=32):
        skip = x
        x = extract_feature_unet(x, reuse,scope,layers=10,ch=32)
        x = conv_layer( x, ch=self.channels,filter_size=7,use_bias=True,name='conv_final',reuse=reuse)
        y = tf.add( x, skip)   
        return y

    def build_uniform_degrade_net(self,x, reuse,scope,training,layers=10,ch=32):
        skip = x
        x = extract_feature_unet(x, reuse,scope,layers=10,ch=32)
        kernel = conv_layer( x, ch=15*15,filter_size=3,use_bias=True,name='conv_final',reuse=reuse)
        kernel = tf.reduce_mean( kernel, axis=[1,2])
        kernel = tf.transpose( tf.reshape( kernel, [-1,15,15,1]),[1,2,0,3])

        pad_sze = 7
        skip = tf.pad(skip,[[0,0],[pad_sze,pad_sze],[pad_sze,pad_sze],[0,0]],'SYMMETRIC')
        skip   = tf.transpose( skip ,(3,1,2,0))
        out = tf.nn.depthwise_conv2d(skip,kernel,[1,1,1,1], padding='VALID')
        out = tf.transpose( out ,(3,1,2,0))
        return out, kernel

    def degrade_net(self):
        border = 7
        self.degrade_model_path = os.path.join('.//sr_degrade', 'checkpoints_' + 'x%s_'%self.factor + self.degrade_model_name ) 
        make_dir(self.degrade_model_path)
        
        global_step  = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        train_input  = tf.placeholder(tf.float32, shape=(None,None,None, self.channels))
        train_label  = tf.placeholder(tf.float32, shape=(None,None,None, self.channels))
        if 'kpn' in self.degrade_model_name:
            train_output,self.kpn = degrade_kpn(train_input, False, 'degradenet',layers=20,ch=32)
        else:
            train_output = self.direct_degrade_unet(train_input, False, 'degradenet',True,layers=20,ch=32)
        residual     = train_output - train_label
        loss = tf.reduce_sum(tf.nn.l2_loss( residual[:,border:-border,border:-border,:]  ))

        all_vars = tf.trainable_variables()
        loss += tf.nn.l2_loss(all_vars[0])*1e-5

        lr = tf.train.piecewise_constant(global_step, boundaries=[6000],values=[1e-03,1e-04])
        optimizer = tf.train.AdamOptimizer(learning_rate=lr) 
        opt   = optimizer.minimize(loss,  global_step=global_step, var_list= all_vars)

        self.saver = tf.train.Saver(all_vars,global_step, max_to_keep=0)
        # training
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            tf.initialize_all_variables().run()
            ckpt_name,start,model_lists = check_checkpoints(self.degrade_model_path)
            if ckpt_name:
                self.saver.restore(sess,ckpt_name)
            # for evaluation only
            # test_psnr = []
            # test_ssim = []
            # for epoch, ckpt_name in enumerate(model_lists):
            #     self.saver.restore(sess,ckpt_name[:-5])
            #     th_psnr,th_ssim = self.evaluate_degrade_net(sess,'test',train_output,train_input,save=False)
            #     test_psnr.append(th_psnr)
            #     test_ssim.append(th_ssim)
            #     print('results for epoch %02d'%(epoch))
            #     print(test_psnr)
            #     print(test_ssim)
            #     print('best psnr: %.2f'%(np.array(test_psnr).max()))
            #     print('best ssim: %.4f'%(np.array(test_ssim).max()))

            self.train_list,self.train_files = get_sr_patches_lists(self.factor,sze=self.IMG_SIZE[0],stride = self.IMG_SIZE[0]) 

            train_loss = np.zeros(self.degrade_EPOCH)
            test_psnr = []
            for epoch in  range(start, self.degrade_EPOCH):
                error = 0
                shuffle(self.train_list)
                for step in range(len(self.train_list)//self.BATCH_SIZE):
                    offset = step*self.BATCH_SIZE
                    
                    hr_data,lr_data = self.get_image_batch( self.train_list,  offset, self.BATCH_SIZE)
                    hr_data,lr_data = data_augment( [hr_data,lr_data ])

                    feed_dict = {train_input:hr_data, train_label:lr_data} # feed_dict: dict
                    s = time.time()
                    th_lr,_,l2,output, g_step = sess.run([lr,opt, loss, train_output, global_step],
                                                    feed_dict=feed_dict)
                    t = time.time()
                    if (g_step-1) % 50 == 0:
                        print( "[epoch %d: %d/%d] mse loss %.4f,  psnr %.2f,  time %.4f, "%( epoch,step, 
                                            len(self.train_list)//self.BATCH_SIZE,np.sum(l2)/self.BATCH_SIZE,
                                            20*math.log10(math.sqrt(self.IMG_SIZE[0]**2*self.channels/np.sum(l2)*self.BATCH_SIZE)),
                                            (t-s)))
                    error = error + l2
                # if (epoch +1)% 50 == 0:
                
                self.saver.save(sess, self.degrade_model_path+"/conv_1l_epoch_%03d.ckpt" % epoch)
                th_test_psnr,_ = self.evaluate_degrade_net(sess,'test',train_output,train_input,save=False)
                print('kernel generation stage PSNR: %.2f\t for epoch : %08d'%(th_test_psnr, epoch))
                test_psnr.append( th_test_psnr )
                    
                train_loss[epoch] = error / len(self.train_list)
                print( "[epoch %d] l2 loss %.4f\t"%( epoch, train_loss[epoch] ))
                plt.plot(train_loss)
                plt.savefig(os.path.join( self.degrade_model_path,'train_loss.png'))
                plt.close()
                # print('train loss:' , train_loss)
                
            trainset_psnr,_ = self.evaluate_degrade_net(sess,'train',train_output,train_input,save=True)
            print('training set degrade model psnr: ')
            print(trainset_psnr)

            test_psnr = np.array( test_psnr )
            print('train loss:')
            print( train_loss )
            print( 'test psnr:')
            print(test_psnr )
            print( 'max test psnr:')
            print( test_psnr.max())
        return 0

    def evaluate_degrade_net(self,sess,tag,train_output,train_input,save=False):
        # generating synthetic diffraction images
        prob = tf.placeholder(tf.float32)
        border = 7
        save_folder = os.path.join( self.degrade_model_path,'vis')
        folder = ['.//realsr_database//RealSR_ycc//Canon//Train','.//realsr_database//RealSR_ycc//Nikon//Train']
        train_save_folder = os.path.join('.//realsr_database//RealSR_ycc//synthetic_lr',self.factor)
        make_dir( save_folder )
        if tag == 'train':
            validate_lists = self.train_files
        elif tag == 'test':
            validate_lists = self.test_lists
        self.tag = tag
        print('evaluation on %s set'%tag)
        
        # generating images
        psnr = []
        ssim = []
        print('evaluation the synthetic HR psnr, #imgs:', len(validate_lists))
        if tag == 'test':
            for i_img in range( len( validate_lists)):
                hr_name,lr_name = validate_lists[i_img]
                hr_img  = read_img( hr_name )
                lr_img  = read_img( lr_name )
                
                syn_lr_img = test_patch(hr_img, train_input, train_output,sess,prob,test_sze=600,stride=500)
                # syn_lr_img = sess.run(train_output,feed_dict={train_input:hr_img[np.newaxis,:,:,np.newaxis]})
                syn_lr_img = np.squeeze(np.array(syn_lr_img))
                
                psnr.append( cal_psnr(syn_lr_img[border:-border,border:-border],lr_img[border:-border,border:-border]))
                # ssim.append( compare_ssim( syn_lr_img[border:-border,border:-border], 
                #                             lr_img[border:-border,border:-border],multichannel=False))
                ssim.append( 0 )
                if False:
                    name = lr_name[-17:]
                    print('saving synthetic lr image on SR trainset for visualization, No. ',i_img)
                    save_name = os.path.join(train_save_folder,name.replace('.mat','_'+self.degrade_model_name+'.png'))
                    save_png(save_name,syn_lr_img)
                    save_name = os.path.join(train_save_folder,name.replace('.mat','_'+self.degrade_model_name+'.mat'))
                    scipy.io.savemat(save_name,{'img':syn_lr_img})
            print('mean psnr for this epoch: %.2f'%( np.array(psnr).mean() ))
            print('mean ssim for this epoch: %.4f'%( np.array(ssim).mean() ))

        if save:
            # for hrset in ['diffract','wesaturate','flickr2k']:
            for hrset in ['diffract']:
                print('generating synthetic LR imgs on %s dataset...'%(hrset))
                folder = os.path.join('.//realsr_database',hrset,'HR')  # diffract
                save_folder = os.path.join('.//realsr_database',hrset,'synthetic',self.factor)
                imglists = glob.glob(os.path.join(folder,'*.mat'))
                print('saving training number: #%d...'%(len(imglists)))

                for i_img in range(len(imglists)):
                    hr_name = imglists[i_img]
                    name = hr_name[-8:]
                    hr_img = scipy.io.loadmat(hr_name)['img']
                    h, w = hr_img.shape[:2]
                    hr_img  = np.pad( hr_img, ((0,h%2),(0,w%2)),'reflect')
                    syn_lr_img = test_patch(hr_img, train_input, train_output,sess,prob,test_sze=192,stride=144)
                    # syn_lr_img = sess.run(train_output,feed_dict={train_input:hr_img[np.newaxis,:,:,np.newaxis]})
                    
                    syn_lr_img = syn_lr_img.squeeze()
                    syn_lr_img = syn_lr_img[:h,:w]
                    save_name = os.path.join(save_folder,name.replace('.mat','_'+self.degrade_model_name+'.png'))
                    save_png(save_name,syn_lr_img)
                    
        
        return np.array(psnr).mean(),np.array(ssim).mean()


    def uniform_degrade_net(self):
        border = 7
        self.degrade_model_path = os.path.join('.//sr_degrade', 'checkpoints_' + 'x%s_'%self.factor + self.degrade_model_name ) 
        make_dir(self.degrade_model_path)
        
        global_step  = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        train_input  = tf.placeholder(tf.float32, shape=(None,None,None, self.channels))
        train_label  = tf.placeholder(tf.float32, shape=(None,None,None, self.channels))
        train_output, kernel = self.build_uniform_degrade_net(train_input, False, 'degradenet',True,layers=20,ch=32)
        self.uniform_kernel = kernel

        residual     = train_output - train_label
        loss = tf.reduce_sum(tf.nn.l2_loss( residual[:,border:-border,border:-border,:]  ))

        all_vars = tf.trainable_variables()
        loss += tf.nn.l2_loss(all_vars[0])*1e-5

        lr = tf.train.piecewise_constant(global_step, boundaries=[6000],values=[1e-03,1e-04])
        optimizer = tf.train.AdamOptimizer(learning_rate=lr) 
        opt   = optimizer.minimize(loss,  global_step=global_step, var_list= all_vars)

        self.saver = tf.train.Saver(all_vars,global_step, max_to_keep=0)
        # training
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            tf.initialize_all_variables().run()
            ckpt_name,start,model_lists = check_checkpoints(self.degrade_model_path)
            if ckpt_name:
                self.saver.restore(sess,ckpt_name)

            self.train_list,self.train_files = get_sr_patches_lists(self.factor,sze=self.IMG_SIZE[0],stride = self.IMG_SIZE[0]) 

            train_loss = np.zeros(self.degrade_EPOCH)
            test_psnr = []
            for epoch in  range(start, self.degrade_EPOCH):
                error = 0
                shuffle(self.train_list)
                for step in range(len(self.train_list)//self.BATCH_SIZE):
                    offset = step*self.BATCH_SIZE
                    
                    hr_data,lr_data = self.get_image_batch( self.train_list,  offset, self.BATCH_SIZE)
                    hr_data,lr_data = data_augment( [hr_data,lr_data ])

                    feed_dict = {train_input:hr_data, train_label:lr_data} # feed_dict: dict
                    s = time.time()
                    th_lr,_,l2,output, g_step = sess.run([lr,opt, loss, train_output, global_step],
                                                    feed_dict=feed_dict)
                    t = time.time()
                    if (g_step-1) % 50 == 0:
                        print( "[epoch %d: %d/%d] mse loss %.4f,  psnr %.2f,  time %.4f, "%( epoch,step, 
                                            len(self.train_list)//self.BATCH_SIZE,np.sum(l2)/self.BATCH_SIZE,
                                            20*math.log10(math.sqrt(self.IMG_SIZE[0]**2*self.channels/np.sum(l2)*self.BATCH_SIZE)),
                                            (t-s)))
                    error = error + l2
                # if (epoch +1)% 50 == 0:
                
                self.saver.save(sess, self.degrade_model_path+"/conv_1l_epoch_%03d.ckpt" % epoch)
                th_test_psnr,_ = self.evaluate_degrade_net(sess,'test',train_output,train_input,save=False)
                print('kernel generation stage PSNR: %.2f\t for epoch : %08d'%(th_test_psnr, epoch))
                test_psnr.append( th_test_psnr )
                    
                train_loss[epoch] = error / len(self.train_list)
                print( "[epoch %d] l2 loss %.4f\t"%( epoch, train_loss[epoch] ))
                plt.plot(train_loss)
                plt.savefig(os.path.join( self.degrade_model_path,'train_loss.png'))
                plt.close()
                # print('train loss:' , train_loss)
                
            trainset_psnr,_ = self.evaluate_degrade_net(sess,'train',train_output,train_input,save=True)
            print('training set degrade model psnr: ')
            print(trainset_psnr)

            test_psnr = np.array( test_psnr )
            print('train loss:')
            print( train_loss )
            print( 'test psnr:')
            print(test_psnr )
            print( 'max test psnr:')
            print( test_psnr.max())
        return 0

    def syn_test(self,model_name,out_path=None):  
        if not out_path: 
            out_path = os.path.join('./test_vis','degradation','sr_x%s'%self.factor)
        self.model_path = os.path.join('./models','sr_x'+self.factor,'checkpoints_kpn_vdsr_syn') 
        self.model_path = os.path.join('./models','sr_x'+self.factor,'checkpoints_r_k15n8_vdsr_syn_real')

        save_tag = '_dml_vdsr.png'
        save  = False
        scale = int(self.factor)
        ##########
        model_list = get_model_lists(self.model_path)
        _,_,model_list = check_checkpoints(self.model_path)

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
            self.test_input  = tf.placeholder(tf.float32, shape=(None, None, None, self.channels))
            self.test_gt  = tf.placeholder(tf.float32, shape=(None, None, None, self.channels))
            lpips_index   = lpips_tf.lpips(tf.tile(self.test_input,(1,1,1,3)), tf.tile(self.test_gt,(1,1,1,3)),
                            model='net-lin', net='alex')
            self.prob        = tf.placeholder(tf.float32)
            if self.mode == 'vdsr':
                self.test_output = self.build_vdsr(x=self.test_input, reuse=self.reuse, scope='sr_unet',training = False)
            elif self.mode == 'rcan':
                self.test_output,_ = build_rcan(x=self.test_input, reuse=self.reuse,ch=1)
            self.saver = tf.train.Saver()  
            
            test_lists = get_syn_test_lists(self.dataset)
            psnr = np.zeros([len(model_list),len(test_lists)])
            ssim = np.zeros([len(model_list),len(test_lists)])
            lpips = np.zeros([len(model_list),len(test_lists)])
            best_psnr = 0
            best_lpips = 1
            
            for i_model in range(len(model_list)):
                model_ckpt = model_list[-1-i_model]
                epoch = int(model_ckpt.split('epoch_')[-1].split('.')[0])
                print( "real sr Testing state for epoch %03d"%(epoch))
                
                self.saver.restore(sess, model_ckpt.split('.meta')[0])

                # prepare synthetic dataset
                for i_group in range(len(test_lists)):
                    hr_name = test_lists[i_group]

                    hr_img = read_img( hr_name)
                    if len(hr_img.shape) > 2:
                        hr_img = rgb2ycbcr_matlab( hr_img)[:,:,0]
                    
                    h,w = hr_img.shape[:2]
                    lr_img = imresize( imresize(hr_img,output_shape=(h//scale, w//scale)),output_shape=(h,w))
                    name = hr_name[-17:]

                    start_t = time.time()
                    if self.mode == 'rcan':
                        output = test_patch(lr_img, self.test_input, self.test_output,sess,self.prob)
                    elif self.mode == 'vdsr':
                        output = sess.run(self.test_output,feed_dict={self.test_input: lr_img[np.newaxis,:,:,np.newaxis],
                                                        self.prob:1.0 })
                    output = np.squeeze( np.array(output) )
                    end_t   = time.time()
                    psnr[i_model,i_group] = cal_psnr(output,hr_img,scale)
                    lpips[i_model,i_group] = sess.run(lpips_index, feed_dict={self.test_input:output[np.newaxis,:,:,np.newaxis], 
                                            self.test_gt: hr_img[np.newaxis,:,:,np.newaxis]})
                    # ssim[i_model,i_group] = compare_ssim(output,hr_img,multichannel=False)
                    ssim[i_model,i_group] = 0
                    print( "sr testing img:%02d,  time %.2f psnr: %.2f\t"%(i_group,end_t-start_t,psnr[i_model,i_group]))
                    if save:
                        output_rgb = ycbcr2rgb_matlab( np.transpose( np.array( 
                                    [output,lr_img_ycc[:,:,1],lr_img_ycc[:,:,2]]),(1,2,0)) )
                        save_png( os.path.join(out_path,name.replace('.png',save_tag)), output_rgb)

                if psnr[i_model,:].mean() > best_psnr:
                    best_psnr  = psnr[i_model,:].mean()
                    best_ckpt  = model_ckpt

                if lpips[i_model,:].mean() <best_lpips:
                    best_lpips  = lpips[i_model,:].mean()
                print( "this psnr %.4f\t best psnr %.4f\t "%(psnr[i_model,:].mean(),best_psnr ))
                print( "this lpips %.4f\t best lpips %.4f\t "%(lpips[i_model,:].mean(),best_lpips ))
                print( "best ssim %.4f\t "%(ssim[i_model,:].mean() ))
                print( self.model_path ) 
                print(np.mean(lpips,-1))
            print( self.model_path )        
            print( "best psnr %.4f\t "%(psnr.mean(1).max()))

        return psnr
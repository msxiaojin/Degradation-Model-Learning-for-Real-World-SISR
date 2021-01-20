import numpy as np
import tensorflow as tf
import lpips_tf,glob,os,scipy

def read_color_img( filename):
    return scipy.misc.imread(filename).astype('float32')/255 

batch_size = 1
image_shape = (batch_size, None, None, 3)
image0_ph = tf.placeholder(tf.float32)
image1_ph = tf.placeholder(tf.float32)

distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')

folder = 'E://DepthOfField//test_vis//sr_x4//rgb'
gt_lists = glob.glob(os.path.join(folder,'*HR.png'))
distance = []
with tf.Session() as sess:
    for i_file in range(len(gt_lists)):
        gt_name = gt_lists[i_file]
        img_name = gt_name.replace('HR.png','LR4_rcan_real.png')
        label  = read_color_img( gt_name)
        output = read_color_img( img_name)
        d = sess.run(distance_t, feed_dict={image0_ph: label, image1_ph: output})
        distance.append(d)

print(np.array(distance).mean())
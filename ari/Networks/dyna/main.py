"""
To do:
1) Error in amin,amax part of if dict display? Correct axis?
2) Archive old version?
"""

# Main

# Author: Ari Herman

import os
import sys
sys.path.insert(0,"../../Classes")
sys.path.insert(0,"../")
import numpy as np
import tensorflow as tf
import pickle
from NeuralData import *
from NeuralAnimation import *
from util import *
from layers import *
import matplotlib
matplotlib.pyplot.switch_backend('agg')
from matplotlib import pyplot as plt
from params import * # Get parameters from external file
import time
np.random.seed(1111)
tf.random.set_seed(1111)

M = int(np.sqrt(n_features))
scale = 1./np.sqrt(np.prod(filter_size))
n_steps = t_steps_per_second*seconds_per_image
steps_per_image = n_steps
t_step = 1./t_steps_per_second
lam = u_decay_rate*t_step
dtype=tf.float32

# Create output directory and log files
#output_dir = sys.argv[1]
output_dir = "output"
os.system("mkdir -p "+output_dir)
os.system("touch "+output_dir+"/run_log.txt")
os.system("touch "+output_dir+"/error_log.txt")
run_log = open(output_dir+"/run_log.txt","w",1)
error_log = open(output_dir+"/error_log.txt","w",1)

# Load data
if dataset == 'imagenet':
    data = imageNetData("../../../datasets/ImageNet")
    def getImage(k):
        im = scale*(data.getBatch(k)) # Centers data
        return im

elif dataset == 'cifar':
    data_path = "../../../datasets/cifar-10-batches-py/"
    def getImage(k):
        with open(data_path+'data_batch_'+str(k+1), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        data = dict[b'data'].reshape((-1,3,32,32))
        data = np.moveaxis(data,[1,2,3],[3,1,2])
        data = (1./np.sqrt(3*16**2))*data/255.
        return data

if activation_type == "soft":
    a = SoftLayer((n_images,image_size[0]//stride,image_size[1]//stride,n_features),lam)
    E = IdentityLayer((n_images,)+image_size)
elif activation_type == "hard":
    a = HardLayer((n_images,image_size[0]//stride,image_size[1]//stride,n_features),lam)
    E = IdentityLayer((n_images,)+image_size)

###################
# Computation graph
###################
'''
This Network is implements a variant of the algorithm described in Replicating Kernels with a Short Stride Allows Sparse Reconstructions wiith Fewer Independent Kernels (Shultz 2014).  The key differences are that the bottom up and top down weights do not share values and the error layer is updated dynamically (like the sparse layer).
'''
# Weights
if load:
    initial_up_weights = np.load(output_dir+'/W_up.npy')
    initial_down_weights = np.load(output_dir+'/W_down.npy')
    W_up = tf.Variable(initial_up_weights, dtype=dtype, name="bottomw_up_active_weights")
    W_down = tf.Variable(initial_down_weights, dtype=dtype, name="top_down_active_weights")

else:
    W_up = getRandomArray(filter_size+(n_features,),sparsity=0.1,normalize=(0,1,2))
    W_down = getRandomArray(filter_size+(n_features,),sparsity=0.1,normalize=(0,1,2))

# Image
image = tf.Variable(np.zeros((n_images,)+image_size),dtype=dtype,name="image")
activity = tf.Variable(np.zeros(n_features),dtype=dtype)

#################
# Setup animation
#################
ani_images = lcaAnimation()
ani_images.setColorMap(None)
ani_images.setMinMax(0.0,1.0)
ani_images.setDataShape((image_size))
ani_images.setDisplaySize((2,2))
ani_images.initFig()


###################
# Main training loop
####################

i = 0
for n in range(n_batch):
    IM = getImage(n)
    for k in range(batch_size//n_images):
        run_log.write("\nBatch " + str(i) + "\n")
        if i%2==0:
            # Display features
            most_active = (-activity.numpy()).argsort()[:n_features]
            dictionary = W_up.numpy()
            dictionary = np.rollaxis(dictionary[:,:,:,most_active],3)
            dictionary -= np.amin(dictionary,axis=(0,1,2),keepdims=True)
            dictionary /= np.amax(dictionary,axis=(0,1,2),keepdims=True)
            fig0,ax = plt.subplots(M,M,figsize=(5,5))
            for b in range(M):
                for c in range(M):
                    ax[b,c].set_axis_off()
                    ax[b,c].imshow(1-dictionary[M*b+c],vmin=0,vmax=1,interpolation="none")
                    # ax[b,c].imshow(1-dictionary[M*b+c],interpolation="none")
            fig0.savefig(output_dir+"/dict")
            np.save(output_dir+'/W_up',W_up)
            np.save(output_dir+'/W_down',W_down)

        # Increment counter
        i += 1
        tic = time.time()
        # percent_recon_error,ana = runBatch(IM[n_images*k:n_images*(k+1)].reshape((-1,)+image_size),steps_per_image)
        IMAGE = IM[n_images*k:n_images*(k+1)].reshape((-1,)+image_size)
        image.assign(IMAGE) # Load a new image
        a.setState(lam*np.ones((n_images,image_size[0]//stride,image_size[1]//stride,n_features)))
        E.setState(np.zeros((n_images,)+image_size))
        for step in range(steps_per_image):
            '''
            All updates (activities AND weights) are performed at each time step.
            Note that the updates appear to be "backwards"; this is to ensure values
            at time, t, depend only on values at time t-1.
            '''
            #########
            # Updates
            #########
            I = alpha*conv2d(E.activity(),W_up,stride) + a.activity()
            recon = conv2dTranspose(a.activity(),W_down,(n_images,)+image_size,stride)
            IE = image - recon

            E.setInput(IE)
            a.setInput(I)

            E.updateState(tau_E) # Set tau_E = 1 to make error equal to image - recon
            a.updateState(tau)

            E.updateActivity()
            a.updateActivity()

            dW = learning_rate*hebb(E.activity(),a.activity(),filter_size+(n_features,),stride) #Original

            # Experimental
            W_down.assign_add(dW)
            W_up.assign_add(dW)
            W_up.assign(tf.nn.l2_normalize(W_up,[0,1,2]))
            W_down.assign(tf.nn.l2_normalize(W_down,[0,1,2]))

            # Print total energy
            if i%1==0 and step%20==0:
                # #Compute energy
                L = 0.5*tf.sqrt(tf.reduce_sum(E.activity()**2)) + lam*tf.reduce_sum(a.activity())
                run_log.write("\tEnergy: "+str(L.numpy())+"\n")

        im_val = tf.nn.l2_loss(image)
        err_val = tf.nn.l2_loss(IE)
        percent_recon_error = 100.0*err_val/im_val

        ana = tf.reduce_mean(tf.reduce_sum(tf.cast(a.activity()>1e-25,dtype),[1,2,3]))

        toc = time.time()
        run_log.write("Batch training time: "+str(toc-tic)+" seconds\n")
        avg_nodes_active = ana.numpy()
        run_log.write( "Active nodes at time " + str(i) + " :" + str(avg_nodes_active) + "/" + str(int(image_size[0]**2*n_features/stride**2)) + " = " + format(100.0*avg_nodes_active/(image_size[0]**2*n_features/stride**2),'.2f') + "%\n")
        RECON = recon.numpy()/scale
        ani_images.animate(RECON.reshape((-1,)+image_size),show=False,save=True,fname=output_dir+"/recon")
        run_log.write("Percent reconstruction error: " + format(percent_recon_error.numpy(),'.2f') + "%\n")
        activity.assign_add(tf.reduce_sum(tf.cast(a.activity()>1e-8,dtype),axis=(0,1,2)))


run_log.close()
error_log.close()

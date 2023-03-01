# Tensorflow functions

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def conv2d(x,w,stride,conv_type="SAME"):
    return tf.nn.convolution(x,w,strides=[stride,stride],padding=conv_type)

def conv2dTranspose(y,w,out_shape,stride,conv_type="SAME"):
    return tf.nn.conv2d_transpose(y,w,out_shape,[1,stride,stride,1],padding=conv_type)

def hebb(x,y,filter_shape,stride,conv_type="SAME",local=False):
    '''
    If local is False, w should be a normal convolutional kernel.  Otherwise, local should be a 4 dimensional array        containing a different kernel for each neuron in the post.
    '''
    patches = tf.image.extract_patches(x,[1,filter_shape[0],filter_shape[1],1],[1,stride,stride,1],[1,1,1,1],conv_type)
    if local:
        return tf.einsum('ijkl,ijkm->jklm',patches,y)
    else:
        N = x.get_shape().as_list()[0]
        return tf.reshape(tf.einsum("ijkl,ijkm->lm",patches,y),filter_shape)/(2*N)

def hopfieldEnergy(x,y,W):
    return 0.5*tf.reduce_sum(tf.transpose(x)@y*W)

def getRandomArray(size,sparsity=0.05,symmetric=False, normalize = None, zero_diag = True, mn=-0.5,mx=0.5):
    w_init = (mx-mn)*tf.random.uniform(size)+mn
    w_init *= tf.cast(tf.random.uniform(size)<sparsity,tf.float32) # Sparse connections*= tf.random.uniform((layer_size[i],layer_size[j]))<0.1 # Sparse connections
    if not normalize is None:
        w_init /= tf.math.sqrt(tf.reduce_sum(w_init**2,axis=normalize,keepdims=True)+1e-10)
    if symmetric:
        assert(size[0]==size[1])
        w_init = 0.5*(w_init+tf.transpose(w_init)) # Make symmetric
        tf.linalg.set_diag(w_init,tf.zeros((size[0],),dtype=tf.float32)) # Zero diagonal
    return tf.Variable(w_init,dtype=tf.float32)

class ImageGrid(): # Not currently working
    def __init__(self,grid_size,image_size,vmin=0,vmax=1):
        self.vmin,self.vmax=0,1
        self.grid_size = grid_size
        self.image_size=image_size
        self.fig, self.ax = plt.subplots(self.grid_size[0],self.grid_size[1],figsize=(3,3))
        self.text0 = plt.text(-170,-170,"")
        for b in range(self.grid_size[0]):
             for c in range(self.grid_size[1]):
                 self.ax[b,c].imshow(np.squeeze(np.zeros(self.image_size)),vmin=self.vmin,vmax=self.vmax,interpolation="nearest")
                 self.ax[b,c].set_axis_off()
    def setDataShape(self,data_shape):
        self.data_shape = data_shape
    def setColorMap(self,colormap):
        self.cmap = colormap
    def setMinMax(self,vmin,vmax):
        self.vmin,self.vmax = vmin,vmax
    def set(self,arr,show=True,save=False,fname="",normalize=False):
        normalized_arr = arr
        if normalize:
            normalized_arr = arr - np.amin(arr,axis=(0,1,2),keepdims=True)
            normalized_arr /= np.amax(normalized_arr,axis=(0,1,2),keepdims=True)
        for b in range(self.grid_size[0]):
             for c in range(self.grid_size[1]):
                 scaled_image = np.squeeze(normalized_arr[self.grid_size[0]*b+c])
                 self.ax[b,c].set_data(scaled_image)
    def show(self):
        self.text0.set_text("")
        self.fig.draw()
        self.fig.pause(0.01)
    def save(self,fname):
        self.text0.set_text("")
        self.fig.savefig(fname+".png")

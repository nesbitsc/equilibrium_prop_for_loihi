# Data storage object: 

# Loads and stores data in standardized format to be used by NeuralNet.py and NeuralAnimation.py

# Author: Ari Herman

import numpy as np
#from sklearn.preprocessing import normalize,Normalizer
#from sklearn.preprocessing import scale as Scale
#from skimage import io
#import imageio
#import pickle
#import glob
       
class Data():
    data_path = "" # Absolute path to dataset
    name = "" # Name of dataset (e.g. cifar, mnist)
    shape = -1 # Shape of a single dataset element (e.g. a single image)
    n_batch = -1 # Number of batches that the dataset will be loaded with
    n_classes = -1 # Number of classes in the dataset 
    data = None # Stores dataset or current batch
    batch_size = -1 # Size of batches to be loaded
    shift = None
    scale = None

    def __init__(self, data_path):
        self.data_path = data_path 
    
    def getDataShape(self):
        return self.shape

    def getNumClasses(self):
        return self.n_classes

    def getNumBatches(self):
        return self.n_batch

    def getBatchSize(self,scale=1.0):
        return self.batch_size

    def setData(self,k):
        raise notImplementedError

#    def getWhiteMatrix(self,batch):
#        flat_batch = batch.reshape((self.batch_size,-1)).T # Flatten images
#        cov = np.cov(flat_batch,rowvar=True) # Get covariance matrix
#        U,S,V = np.linalg.svd(cov) # Singular value decomposition
#        zca = np.dot(U,np.dot(np.diag(1./np.sqrt(S+1e-8)),U.T)).T # ZCA whitening matrix
#        return zca
#        
#    def whiten(self,batch,zca):
#        return np.dot(batch.reshape((self.batch_size,-1)),zca).reshape(np.shape(batch))
#
#    def unwhiten(self,batch,zca): 
#        return np.dot(batch.reshape((self.batch_size,-1)),np.linalg.inv(zca)).reshape(np.shape(batch))

    def getBatch(self,k,scale=1.0,cnt='01',subsample=None):
        self.setData(k)
        if subsample != None:
            subsample = self.data[:,(self.shape[0]-subsample[0])//2:subsample[0]+(self.shape[0]-subsample[0])//2,(self.shape[1]-subsample[1])//2:subsample[1]+(self.shape[1]-subsample[1])//2,:]
        else:
            subsample = self.data
        if cnt == '01':
            return scale*(subsample - self.shift)/self.scale
        elif cnt == 'center':
            avg = np.mean(subsample,axis=0,keepdims=True)
            sd = np.std(subsample,axis=0,keepdims=True)+1e-10
            return scale*(subsample-avg)/sd

class mnistData(Data):
    def __init__(self, data_path):

        super().__init__(data_path) 
        self.name = "mnist" 
        self.shape = (28,28,1)
        self.n_batch = 6
        self.n_classes = 10
        self.batch_size = 10000
        self.scale = 255.0
        self.shift = 0.0
    def setData(self,k):
        self.data = np.load(self.data_path + "/Mnist_" + format(k,'01d') + '.npy')

class cifarData(Data):
    def __init__(self, data_path):

        super().__init__(data_path) 
        self.name = "cifar" 
        self.shape = (32,32,3)
        self.n_batch = 5
        self.n_classes = 10
        self.batch_size = 10000
        self.scale = 255.0
        self.shift = 0.0
    def setData(self,k):
        self.data = np.load(self.data_path + "/Cifar_" + format(k,'01d') + '.npy')

class imageNetData(Data):
    def __init__(self, data_path):

        super().__init__(data_path) 
        self.name = "imagenet" 
        self.shape = (256,256,3)
        self.n_batch = 755
        self.n_classes = 1000
        self.batch_size = 1500
        self.scale = 255.0
        self.shift = 0.0
    def setData(self,k):
        self.data = np.load(self.data_path + "/ImageNet_" + format(k,'03d') + '.npy')

class imageNetPatchData(Data):
    def __init__(self, data_path):

        super().__init__(data_path) 
        self.name = "imagenetpatch" 
        self.shape = (16,16,3)
        self.n_batch = 151
        self.n_classes = 1000
        self.batch_size = 7500
        self.scale = 255.0
        self.shift = 0.0
    def setData(self,k):
        self.data = np.load(self.data_path + "/ImageNetPatch_" + format(k,'03d') + '.npy')

class grayPatchData(Data):
    def __init__(self, data_path):

        super().__init__(data_path) 
        self.name = "graypatch" 
        self.shape = (16,16,1)
        self.n_batch = 151
        self.n_classes = 1000
        self.batch_size = 7500
        self.scale = 255.0
        self.shift = 0.0
    def setData(self,k):
        self.data = np.load(self.data_path + "/GrayPatch_" + format(k,'03d') + '.npy')

class youTubeData(Data):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.name = "youtube"
        self.shape = (128,256,256,3)
        self.n_batch = 1410
        self.n_classes = 1e10
        self.batch_size = 1
        self.scale = 255.0
        self.shift = 127.5
    def setData(self,k):
        batch = []
        for i in range(k*self.batch_size,(k+1)*self.batch_size):
            batch.append( np.load(self.data_path + "/YouTube_" + format(i,'04d')+'.npy') )
        self.data = np.stack(batch,axis=0)
    def getBatch(self,k,scale=1.0):
        self.setData(k)
        return scale*(self.data-self.shift)/self.scale

class movieClipData(Data):
    def __init__(self, data_path):

        super().__init__(data_path) 
        self.name = "movieclips" 
        self.shape = (4,256,256,3)
        self.n_batch = 96
        self.n_classes = 1e20
        self.batch_size = 512
        self.scale = 255.0
        self.shift = 127.5
    def setData(self,k):
        self.data = np.load(self.data_path + "/MovieClip_" + format(k,'02d') + '.npy')[:,:self.shape[0],:,:,:] #.reshape((-1,)+self.shape)
    def getBatch(self,k,scale=1.0):
        self.setData(k)
        return scale*(self.data-self.shift)/self.scale

class youTubeClipData(Data):
    def __init__(self, data_path):
        super().__init__(data_path) 
        self.name = "youtubeclips" 
        self.shape = (4,256,256,3)
        self.n_batch = 29
        self.n_classes = 1e20
        self.batch_size = 256
        self.scale = 255.0
        self.shift = 127.5
    def setData(self,k):
        self.data = np.load(self.data_path + "/YouTubeClip_" + format(k,'02d') + '.npy')[:,:self.shape[0],:,:,:] #.reshape((-1,)+self.shape)
    def getBatch(self,k,scale=1.0):
        self.setData(k)
        return scale*(self.data-self.shift)/self.scale

class movieData(Data):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.name = "movie"
        self.shape = (64,256,256,3)
        self.n_batch = 3076
        self.n_classes = 1e10
        self.batch_size = 1
        self.scale = 255.0
        self.shift = 127.5
    def setData(self,k):
        batch = []
        for i in range(k*self.batch_size,(k+1)*self.batch_size):
            batch.append( np.load(self.data_path + "/Movie_" + format(i,'04d')+'.npy') )
        self.data = np.stack(batch,axis=0)
    def getBatch(self,k,scale=1.0):
        self.setData(k)
        return scale*(self.data-self.shift)/self.scale

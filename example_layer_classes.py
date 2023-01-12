import torch
import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    
    def __init__(self):
        self.__prevIn__ = []
        self.__prevOut__ = []
        
    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn
        
    def setPrevOut(self, out):
        self.__prevOut = out
        
    def getPrevIn(self):
        return self.__prevIn
    
    def getPrevOut(self):
        return self.__prevOut
    
    def backward(self, gradIn):
        return (gradIn @ self.gradient())
    
    @abstractmethod
    def forward(self, dataIn):
        pass
    
    @abstractmethod
    def gradient(self):
        pass

class HardSigmoidLayer(Layer):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)      
        z = torch.clip(self.getPrevIn(), -1, 1)
        self.setPrevOut(z)
        return self.getPrevOut()
        
    def gradient(self): 
        z = (self.getPrevOut() > -1) & (self.getPrevOut() < 1)
        return z

class FullyConnectedLayer(Layer):
    
    def __init__(self, sizeIn, sizeOut):
        super().__init__()
        np.random.seed(0)
        self.__weights = np.random.uniform(-1/np.sqrt(sizeIn),1/np.sqrt(sizeIn),(sizeIn,sizeOut))
        self.__bias = np.random.uniform(-1/np.sqrt(sizeIn),1/np.sqrt(sizeIn),(1,sizeOut))
    
    def getWeights(self):
        return self.__weights
        
    def setWeights(self, weights):
        self.__weights = weights
        
    def getBias(self):
        return self.__bias
        
    def setBias(self, bias):
        self.__bias = bias
        
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(self.getPrevIn() @ self.getWeights() + self.getBias())
        return self.getPrevOut()
        
    def gradient(self):
        return self.getWeights().T
    
    def updateWeights(self, gradIn, eta=0.01):
        dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]
        dJdW = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]
        self.setWeights(self.getWeights() - (eta * dJdW))
        self.setBias(self.getBias() - (eta * dJdb))
        
    def backward(self, gradIn):
        self.updateWeights(gradIn)
        return super().backward(gradIn)

class CrossEntropyLoss():
    
    def eval(self,y,yhat):
        return (-np.sum(y*np.log(yhat+np.finfo(float).eps)))
    
    def gradient(self,y,yhat):
        return (-1 * (y/(yhat+np.finfo(float).eps)))
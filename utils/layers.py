import tensorflow as tf
import numpy as np
from abc import ABC

class Layer(ABC):

    def __init__(self):
        self.__prevIn__ = []
        self.__prevOut__ = []
        self.__state__ = None

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut

    def getState(self):
        return self.__state

    @abstractmethod
    def forward(self, dataIn):
        pass



class HardSigmoidLayer(Layer):

    def __init__(self, size, dtype=tf.float32):
        super().__init__()
        self.state = tf.Variable(tf.zeros(size),dtype=dtype)

    def forward(self,dataIn,epsilon):
        self.state.assign_add(epsilon*dataIn)
        activity = tf.clip_by_value(self.state,0.0,1.0)
        return activity

class

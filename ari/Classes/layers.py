import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):

    def __init__(self):
        # self._state = tf.Variable(size,dtype=dtype)
        self._activity = None
        self._input = None

    def setInput(self,dataIn):
        self._input = dataIn

    def getInput(self):
        return self._input

    def getState(self):
        return self._state

    def setState(self, val):
        self._state.assign(val)

    @abstractmethod
    def updateState(self, dataIn):
        pass

    @abstractmethod
    def updateActivity(self):
        pass

    @abstractmethod
    def activity(self):
        pass

class HardSigmoidLayer(Layer):

    def __init__(self, size: tuple, dtype=tf.float32):
        super().__init__()
        self._state = tf.Variable(tf.zeros(size),dtype=dtype)

    def updateState(self,epsilon, mn = 0.0, mx = 1.0):
        self._state.assign( tf.clip_by_value( self._state*(1-epsilon) + epsilon*self._input, mn, mx))

    def updateActivity(self):
        self._activity = self._state

    def activity(self):
        return self._activity

class HardSigmoid2Layer(Layer):

    def __init__(self, size: tuple, dtype=tf.float32,shift=0.0,mn=0.0,mx=1.0):
        super().__init__()
        self._state = tf.Variable(tf.zeros(size),dtype=dtype)
        self.shift=shift
        self.mn =mn
        self.mx=mx

    def updateState(self,epsilon):
        self._state.assign( self._state*(1-epsilon) + epsilon*self._input )

    def updateActivity(self):
        self._activity = tf.clip_by_value(self._state-self.shift,self.mn,self.mx)

    def activity(self):
        return self._activity


class SoftSigmoidLayer(Layer):

    def __init__(self, size: tuple, dtype=tf.float32):
        super().__init__()
        self._state = tf.Variable(tf.zeros(size),dtype=dtype)
        self._activity = tf.Variable(tf.zeros(size),dtype=dtype)
        self._input = tf.Variable(tf.zeros(size),dtype=dtype)

    def updateState(self,epsilon):
        # self._state.assign_add( epsilon*(self._activity*(1-self._activity)*self._input - self._state) )
        self._state.assign_add( epsilon*(self._input - self._state) ) # This actually works better

    def updateActivity(self):
        self._activity = 1./(1 + tf.exp(-self._state))

    def activity(self):
        return self._activity

class TanhLayer(Layer):

    def __init__(self, size: tuple, dtype=tf.float32):
        super().__init__()
        self._state = tf.Variable(tf.zeros(size),dtype=dtype)
        self._activity = tf.Variable(tf.zeros(size),dtype=dtype)
        self._input = tf.Variable(tf.zeros(size),dtype=dtype)

    def updateState(self,epsilon):
        self._state.assign_add( epsilon*((1-self._activity**2)*self._input - self._state) )
        # self._state.assign_add( epsilon*(self._input - self._state) ) # This actually works better

    def updateActivity(self):
        self._activity = tf.math.tanh(self._state)

    def activity(self):
        return self._activity

class SoftLayer(Layer):

    def __init__(self, size: tuple, lam, dtype=tf.float32):
        super().__init__()
        self._state = tf.Variable(tf.zeros(size),dtype=dtype)
        self._activity = tf.Variable(tf.zeros(size),dtype=dtype)
        self._input = tf.Variable(tf.zeros(size),dtype=dtype)
        self.lam = lam
    def updateState(self,epsilon):
        self._state.assign_add( epsilon*(self._input - self._state) )

    def updateActivity(self):
        self._activity = 1./(tf.math.log(1-self.lam/tf.maximum(self._state,self.lam+1e-20))/tf.math.log(1-self.lam)+1.5)

    def activity(self):
        return self._activity

class HardLayer(Layer):

    def __init__(self, size: tuple, lam, dtype=tf.float32):
        super().__init__()
        self._state = tf.Variable(tf.zeros(size),dtype=dtype)
        self._activity = tf.Variable(tf.zeros(size),dtype=dtype)
        self._input = tf.Variable(tf.zeros(size),dtype=dtype)
        self.lam = lam
    def updateState(self,epsilon):
        self._state.assign_add( epsilon*(self._input - self._state) )
        # self._state.assign_add( epsilon*(self._input - self._state) ) # This actually works better

    def updateActivity(self):
        self._activity = tf.maximum(self._state-self.lam,0.0) #*tf.cast(self._state>self.lam,tf.float32)

    def activity(self):
        return self._activity

class IdentityLayer(Layer):
    def __init__(self, size: tuple, dtype=tf.float32):
        super().__init__()
        self._state = tf.Variable(tf.zeros(size),dtype=dtype)
        self._activity = tf.Variable(tf.zeros(size),dtype=dtype)
        self._input = tf.Variable(tf.zeros(size),dtype=dtype)
    def updateState(self,epsilon):
        # self._state.assign_add( epsilon*(self._input - self._state) )
        self._state.assign( epsilon*self._input +(1-epsilon)*self._state )
        # self._state.assign_add( epsilon*(self._input - self._state) ) # This actually works better

    def updateActivity(self):
        self._activity = self._state

    def activity(self):
        return self._activity

"""
To do:

1) Add biases.
2) Make nice seaborn graphs
3) Add hidden layers

"""
scale = 2

import tensorflow as tf
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os
from params import * # Get parameters from external file
np.random.seed(1111)
tf.random.set_seed(1111)
n_steps = seconds_per_image*t_steps_per_second # Time steps per image
t_step = 1./t_steps_per_second #(s)
input_size=784
hidden_size=64 # 200 # 500
batch_size = 20
target_size = 10
# Learning rates
alpha1 = 0.2 #0.1
alpha2 = 0.1 #0.05
alpha3 = -0.0
alpha={}
alpha['x','h'] = alpha1
alpha['x','y'] = alpha1
alpha['h','h'] = alpha3
alpha['h','y'] = alpha2
beta = 1.0
gamma = 0.0 # Lateral inhibition strength
epsilon = 0.5/scale

activation_type = "sigmoid"

n_features = 64
M=8

layers = ['x','h','y']
connections = [('x','h'),('h','y')]
layer_size = {'x':input_size,'h':64,'y':10}

# Create output directory and log files
#output_dir = sys.argv[1]
output_dir = "output"
os.system("mkdir -p "+output_dir)
os.system("touch "+output_dir+"/run_log.txt")
os.system("touch "+output_dir+"/error_log.txt")
run_log = open(output_dir+"/run_log.txt","w",1)
error_log = open(output_dir+"/error_log.txt","w",1)

data_path = "../../../datasets/"

data = np.genfromtxt(data_path+"mnist_train.csv",delimiter=",",dtype=np.uint8) #Load MNIST data from csv file
# data = np.load(data_path+"mnist.npy") # Load black and white MNIST data

images = data[:,1:]/255.
targets = np.zeros((60000,target_size))
targets[np.arange(60000),data[:,0]]=1.0

# Activation functions
def sigma(X):
    return 1./(1+tf.exp(-X))

def hard(X):
    return tf.clip_by_value(X,0.0,1.0)

def harder(X):
    return tf.clip_by_value(X,0.001,1.0)

def soft(X, lam=0.0008):
    1./(tf.math.log(1-lam/tf.maximum(X,lam+1e-20))/tf.math.log(1-lam)+1.5)

######################
# Create weight arrays
######################
W_init = {}
W = {}
for i,j in connections:
    if i!=j:
        W_init[i,j] = (tf.random.uniform((layer_size[i],layer_size[j]))-0.5)/layer_size[i]
        W_init[i,j] /= tf.math.sqrt(tf.reduce_sum(W_init[i,j]**2,axis=(0,),keepdims=True)+1e-10)
        W[i,j] = tf.Variable(W_init[i,j],dtype=tf.float32)
# Lateral weights
W_init['h','h'] = tf.matmul(tf.transpose(W['x','h']),W['x','h'])
tf.linalg.set_diag(W_init['h','h'],tf.zeros((layer_size['h'],),dtype=tf.float32))
W['h','h'] = tf.Variable(W_init['h','h'],dtype=tf.float32)

# Setup variables
x = tf.Variable(tf.zeros((batch_size,input_size)),dtype=tf.float32) #Input vector
uh = tf.Variable(tf.zeros((batch_size,hidden_size)),dtype=tf.float32) #Potentials
uy = tf.Variable(tf.zeros((batch_size,target_size)),dtype=tf.float32) #Potentials
h0_min = tf.Variable(tf.zeros((batch_size,hidden_size)),dtype=tf.float32) # Local mins for h free
hbeta_min = tf.Variable(tf.zeros((batch_size,hidden_size)),dtype=tf.float32) # Local mins for h with weak clamped target
y0_min = tf.Variable(tf.zeros((batch_size,target_size)),dtype=tf.float32) # Local mins for y
ybeta_min = tf.Variable(tf.zeros((batch_size,target_size)),dtype=tf.float32) # Local mins for y with weak clamped target
d = tf.Variable(tf.zeros((batch_size,target_size)),dtype=tf.float32) # Targets
activity = tf.Variable(np.zeros(hidden_size),dtype=tf.float32) # Activity

I = {layer:0 for layer in layers}
a = {'x':x}

###############
# Training loop
###############
for epoch in range(500):
    for batch in range(60000//batch_size): # Iterate over mini batches
        x.assign(images[batch*batch_size:(batch+1)*batch_size]) # Set input batch
        d.assign(targets[batch*batch_size:(batch+1)*batch_size]) # Set targets

        #########
        # Phase 1
        #########
        for step in range(20*scale):

            #Activations / firing rates
            if activation_type == "sigmoid":
                a['h'] = sigma(uh)
                a['y'] = sigma(uy)
            elif activation_type == "hard":
                a['h'] = hard(uh)
                a['y'] = hard(uy)
            elif activation_type == "harder":
                a['h'] = harder(uh)
                a['y'] = harder(uy)

            #Inputs
            I['h'] = tf.matmul(a['x'],W['x','h'])+tf.matmul(a['y'],tf.transpose(W['h','y']))+gamma*tf.matmul(a['h'],W['h','h']) #Input current
            I['y'] = tf.matmul(a['h'],W['h','y'])#+tf.matmul(a['x'],W['x','y'])#+tf.matmul(a['y'],W['y','y'])

            #Update potentials
            if activation_type == "sigmoid":
                uh.assign_add((a['h']*(1-a['h'])*I['h']-uh)*epsilon)
                uy.assign_add((a['y']*(1-a['y'])*I['y']-uy)*epsilon)
            elif activation_type == "hard":
                uh.assign(hard(uh*(1-epsilon)+I['h']*epsilon))
                uy.assign(hard(uy*(1-epsilon)+I['y']*epsilon))
            elif activation_type == "harder": # Correct derivative??????
                uh.assign(harder(uh*(1-epsilon)+I['h']*epsilon))
                uy.assign(harder(uy*(1-epsilon)+(I['y']+beta*(d-a['y']))*epsilon)) # Correct? Same as in paper???


            # Print total energy
            if batch%10==0 and step%(2*scale)==0:
                # #Compute energy
                E = 0.5*(tf.reduce_sum(uh**2)+tf.reduce_sum(uy**2))
                for i,j in connections:
                    E -= 0.5*tf.reduce_sum(tf.matmul(tf.transpose(a[i]),a[j])*W[i,j])

                run_log.write("\tEnergy: "+str(E.numpy()*100)+"%\n")

        #Save local mins
        h0_min.assign(a['h'])
        y0_min.assign(a['y'])

        # Get Predictions
        predictions = tf.math.argmax(a['y'],axis=1)
        actual = tf.math.argmax(d,axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.math.equal(actual,predictions),tf.float32))

        # Print to log
        if batch%10==0:
            run_log.write("\n\nRunning epoch: "+str(epoch)+" / batch: "+str(batch)+"\n\n")
            run_log.write("Accuracy: "+str(100*accuracy.numpy())+"%\n\n")
            run_log.write("Average nodes active: "+str(np.sum(a['h'].numpy()>0.001)/batch_size)+" out of " + str(hidden_size))

            # # Average nodes active
            # ana = tf.reduce_mean(tf.reduce_sum(tf.cast(h>1e-25,dtype),[1]))
            # avg_nodes_active = ana.numpy()
            # run_log.write( "Active nodes at time " + str(batch) + " :" + str(avg_nodes_active) + "%\n")

        # Update activity
        activity.assign_add(tf.reduce_sum(a['h'],axis=(0,)))


        #########
        # Phase 2
        #########
        for step in range(4*scale):

            #Activations / firing rates
            if activation_type == "sigmoid":
                a['h'] = sigma(uh)
                a['y'] = sigma(uy)
            elif activation_type == "hard":
                a['h'] = hard(uh)
                a['y'] = hard(uy)
            elif activation_type == "harder":
                a['h'] = harder(uh)
                a['y'] = harder(uy)

            # Inputs
            I['h'] = tf.matmul(a['x'],W['x','h'])+tf.matmul(a['y'],tf.transpose(W['h','y']))+gamma*tf.matmul(a['h'],W['h','h'])
            I['y'] = tf.matmul(a['h'],W['h','y'])#+tf.matmul(a['x'],W['x','y'])#+tf.matmul(a['y'],W['y','y'])

            # Update potentials
            if activation_type == "sigmoid":
                uh.assign_add((a['h']*(1-a['h'])*I['h']-uh)*epsilon)
                uy.assign_add((a['y']*(1-a['y'])*I['y']-uy+beta*(d-a['y']))*epsilon)
            elif activation_type == "hard":
                uh.assign(hard(uh*(1-epsilon)+I['h']*epsilon))
                uy.assign(hard(uy*(1-epsilon)+(I['y']+beta*(d-a['y']))*epsilon)) # Correct? Same as in paper???
            elif activation_type == "harder":
                uh.assign(harder(uh*(1-epsilon)+I['h']*epsilon))
                uy.assign(harder(uy*(1-epsilon)+(I['y']+beta*(d-a['y']))*epsilon)) # Correct? Same as in paper???

        #Save local mins
        hbeta_min.assign(a['h'])
        ybeta_min.assign(a['y'])

        ################
        # Update weights
        ################
        dW = {}
        dW['x','h'] = (tf.matmul(tf.transpose(a['x']),hbeta_min)-tf.matmul(tf.transpose(a['x']),h0_min))/(beta*batch_size) # This can be simplified
        # dW['x','y'] = (tf.matmul(tf.transpose(a['x']),ybeta_min)-tf.matmul(tf.transpose(a['x']),y0_min))/(beta*batch_size) # This can be simplified
        # dW['h','h'] = (tf.matmul(tf.transpose(hbeta_min),hbeta_min)-tf.matmul(tf.transpose(h0_min),h0_min))/(beta*batch_size)
        # dW['y','y'] = (tf.matmul(tf.transpose(ybeta_min),ybeta_min)-tf.matmul(tf.transpose(y0_min),y0_min))/(beta*batch_size)
        dW['h','y'] = (tf.matmul(tf.transpose(hbeta_min),ybeta_min)-tf.matmul(tf.transpose(h0_min),y0_min))/(beta*batch_size)
        for i,j in connections:
            if i=='h' and j=='h':
                W[i,j].assign(-tf.matmul(tf.transpose(W['x','h']),W['x','h'])) # Usc LCA rule for lateral
            else:
                W[i,j].assign_add(alpha[i,j]*dW[i,j])
                W[i,j].assign(tf.nn.l2_normalize(W[i,j],[0]))

        ##################
        # Display features
        ##################
        if batch%10==0:
            most_active = (-activity.numpy()).argsort()[:n_features] # Select most active features
            dictionary = W['x','h'].numpy().reshape((28,28,-1)) # Reshape to image shape
            dictionary = np.rollaxis(dictionary[:,:,most_active],2)
            dictionary -= np.amin(dictionary,axis=(1,2),keepdims=True)
            dictionary /= np.amax(dictionary,axis=(1,2),keepdims=True)
            fig0,ax = plt.subplots(M,M,figsize=(5,5))
            for b in range(M):
                for c in range(M):
                    ax[b,c].set_axis_off()
                    ax[b,c].imshow(1-dictionary[M*b+c],vmin=0,vmax=1,cmap='gray',interpolation="none")
            fig0.savefig(output_dir+"/dict")


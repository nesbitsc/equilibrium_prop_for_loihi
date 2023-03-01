# Imagenet
dataset = 'imagenet'
activation_type = "soft"
stride = 2
image_size = (256,256,3)
n_features = 64
filter_size = (16,16,3)
learning_rate = 0.05 #0.01 #0.002 #0.006 # 0.01 was working!
tau = 0.05 #0.02
tau_E = 0.05 #0.02
n_batch = 755
batch_size = 1500
n_images = 20
alpha = 0.05
u_decay_rate = 0.016
t_steps_per_second = 20
seconds_per_image = 30 #50
load = True

"""
r1
tau = 0.02
Seconds per image = 25
learning rate = 0.02
12%
18%

r2 (accidentally deleted, it had mostly good gabors but was kind of monochromatic)
tau = 0.02
Seconds per image = 30
learning rate = 0.01
12%
19%

r3
tau = 0.1
second per image = 15
learning rate = 0.01
6%
10%

r4
tau = 0.1
seconds per image = 15
learning rate = 0.005
5%
10%

r5
tau = 0.1
second per image = 15
learning rate = 0.05
4.4%
4%

r6
tau = 0.2
second per image = 15
learing rate = 0.01
4%
5%

 r6
 tau = 0.01
 seconds per image = 150
learning rate = 0.5

"""

# # Cifar
# dataset = 'cifar'
# activation_type = "hard" #"soft"
# stride = 1
# image_size = (32,32,3)
# n_features = 49 # 64
# filter_size = (8,8,3)
# learning_rate = 0.1 # Try increasing this!
# tau = 0.025
# tau_E = 1 #0.025
# n_batch = 5
# batch_size = 10000
# n_images = 20
# alpha = 0.1
# u_decay_rate = 0.016
# t_steps_per_second = 20
# seconds_per_image = 5 #40
# load = False
# hid_layer_size = 100
# target_size = 10

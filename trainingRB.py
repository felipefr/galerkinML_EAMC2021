#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:37:49 2021
Availabel in: https://github.com/felipefr/galerkinML_EAMC2021.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import myHDF5 as myhd
import utilsTensorflow as mytf

Nlist = [5,10,20,40,60,100]
N = Nlist[5]

X = myhd.loadhd5('train.hd5','param')
Y = myhd.loadhd5('basis.hd5','projections')[:,:N]

scalerY = MinMaxScaler()
scalerY.fit(Y)
Y_t = scalerY.transform(Y)

# Network settings
L = 5  # nb of hidden layers
n = X.shape[1]  # input size
m = Y.shape[1] # output size
# Nneurons = [n] + [max(n,int((m/n)*(i+1)*n/L)) for i in range(L)] + [m] # nb neurons hidden layers
Nneurons = [n] + [25,50,100,max(50,N),max(25,N)] + [m] # nb neurons hidden layers
lr = 1.0e-3 # learning rate
decay = 0.5
EPOCHS = 1000 # epochs to be trained
ratio_val = 0.2 # validation ration

print(Nneurons)

# labelfigs = '_L2_NN10_lr1m3'

# Options activations: tf.nn.{tanh, sigmoid, leaky_relu, relu, linear}

# Building the architecture : L (hidden layers) + input + output layer
layers = [tf.keras.layers.Dense( Nneurons[0], activation=tf.keras.activations.tanh, input_shape=(Nneurons[0],))]
for i in range(L):
    layers.append(tf.keras.layers.Dense( Nneurons[i+1], activation=tf.keras.activations.relu))
layers.append(tf.keras.layers.Dense( Nneurons[-1], activation=tf.keras.activations.linear )) 
# layers.append(tf.keras.layers.Dense( Nneurons[-1], activation=tf.nn.sigmoid)) 


model = tf.keras.Sequential(layers)


# Setting the opmisation algorithm
# Options optimisers: Adadelta, Adagrad, Adam, Adamax, FTRL, NAdam, RMSprop, SGD: 
optimizer= tf.keras.optimizers.Adam(learning_rate = lr)
     
facScale = 10.0e5   
w_l = facScale*(scalerY.data_max_[:N] - scalerY.data_min_[:N])**2.0
w_l = w_l.astype('float32')

decay_lr = tf.keras.callbacks.LearningRateScheduler(mytf.partial2(mytf.scheduler ,lr = lr, decay = decay, EPOCHS = EPOCHS))    

model.compile(loss = mytf.custom_loss_mse(np.sqrt(w_l)), optimizer=optimizer, metrics=[mytf.custom_loss_mse(w_l),'mse','mae'])

# # Fitting 
hist = model.fit(X, Y_t, epochs=EPOCHS, validation_split=ratio_val, verbose=1, callbacks=[mytf.PrintDot(), decay_lr ], batch_size = 32)
model.save_weights('weights_RB{0}'.format(N))
# model.load_weights('weights')


# # Creation of test sample
# Nt = 100
# Xtest = np.random.rand(Nt)
# Ytest = 5.0 + Xtest*Xtest + deltaNoise*np.random.randn(Nt)

# Xtest = Xtest.reshape((Nt,n))
# Ytest = Ytest.reshape((Nt,m))

# Ypred = scalerY.inverse_transform( model.predict(scalerX.transform(Xtest)))

# # Plots
plt.figure(1)
plt.plot(np.array(hist.history['loss']), label = 'train')
plt.plot(np.array(hist.history['val_loss']), label = 'validation')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
# plt.savefig('historic_RB{0}.png'.format(labelfigs))
plt.show()

plt.figure(2)
plt.plot(np.array(hist.history['<lambda>'])/facScale, label = 'train')
plt.plot(np.array(hist.history['val_<lambda>'])/facScale, label = 'validation')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('historic_RB{0}.png'.format(N))
plt.show()

loss = np.array(hist.history['<lambda>']).reshape((-1,1))/facScale
loss_val = np.array(hist.history['val_<lambda>']).reshape((-1,1))/facScale
np.savetxt('historic_RB{0}.txt'.format(N), np.concatenate((loss,loss_val), axis = 1))

# plt.figure(2)
# plt.plot(Xtest[:,0], Ytest[:,0], 'o', label = 'test')
# plt.plot(Xtest[:,0], Ypred[:,0], 'o', label = 'prediction')
# plt.grid()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.savefig('prediction_{0}.png'.format(labelfigs))
# plt.show()
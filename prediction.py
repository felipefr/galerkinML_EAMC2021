import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import myHDF5 as myhd
import utilsTensorflow as mytf

# Building the architecture : L (hidden layers) + input + output layer
def getNNmodel(Nneurons):
    layers = [tf.keras.layers.Dense( Nneurons[0], activation=tf.keras.activations.tanh, input_shape=(Nneurons[0],))]
    for i in range(L):
        layers.append(tf.keras.layers.Dense( Nneurons[i+1], activation=tf.keras.activations.relu))
    layers.append(tf.keras.layers.Dense( Nneurons[-1], activation=tf.keras.activations.linear )) 
    
    model = tf.keras.Sequential(layers)
    return model


Nlist = [5,10,20,40,60,100]

X_train = myhd.loadhd5('train.hd5','param')
u_train = myhd.loadhd5('train.hd5','u')
Y_train = myhd.loadhd5('basis.hd5','projections')[:,:Nlist[-1]]

Vrb = myhd.loadhd5('basis.hd5','basis')[:,:Nlist[-1]]
sig = myhd.loadhd5('basis.hd5','sigma')


X_test = myhd.loadhd5('test.hd5','param')
u_test = myhd.loadhd5('test.hd5','u')
Y_test = u_test @ Vrb 

scalerY = MinMaxScaler()
scalerY.fit(Y_train)
Y_train_t = scalerY.transform(Y_train)
Y_test_t = scalerY.transform(Y_test)

# Network settings
L = 5  # nb of hidden layers
n = X_train.shape[1]  # input size

models = []
for N in Nlist:
    Nneurons = [n] + [25,50,100,max(50,N),max(25,N)] + [N] # nb neurons hidden layers
    print(Nneurons)
    models.append(getNNmodel(Nneurons))
    models[-1].load_weights('weights_RB{0}'.format(N).format(N))


# # Creation of test sample
Y_train_pred = []
Y_test_pred = []

for i in range(len(Nlist)):
    Ytemp = np.concatenate( ( models[i].predict(X_train), np.zeros((len(X_train),Nlist[-1] - Nlist[i]))), axis = 1) 
    Y_train_pred.append(scalerY.inverse_transform(Ytemp))
    Ytemp = np.concatenate( ( models[i].predict(X_test), np.zeros((len(X_test),Nlist[-1] - Nlist[i]))), axis = 1) 
    Y_test_pred.append(scalerY.inverse_transform(Ytemp))

error_DNN_train = np.zeros((len(Nlist), len(X_train)))
error_DNN_train_total = np.zeros((len(Nlist), len(X_train)))
error_DNN_test = np.zeros((len(Nlist), len(X_test)))
error_DNN_test_total = np.zeros((len(Nlist), len(X_test)))

for i in range(len(Nlist)):
    N = Nlist[i]
    for j in range(len(X_train)):
        error_DNN_train[i,j] = np.linalg.norm(Y_train_pred[i][j,:N] - Y_train[j,:N])**2
        error_DNN_train_total[i,j] = np.linalg.norm(Y_train_pred[i][j,:N]@Vrb[:,:N].T - u_train[j,:])**2
        
    for j in range(len(X_test)):
        error_DNN_test[i,j] = np.linalg.norm(Y_test_pred[i][j,:N] - Y_test[j,:N])**2
        error_DNN_test_total[i,j] = np.linalg.norm(Y_test_pred[i][j,:N]@Vrb[:,:N].T - u_test[j,:])**2
        
Ns = 1000
errorPOD = np.zeros(Ns)
for i in range(Ns):
    errorPOD[i] = np.sum(sig[i:]*sig[i:])/Ns
    
plt.figure(1)
# plt.plot(Nlist, np.mean(error_DNN_train, axis = 1), '-o', label = 'DNN train')
plt.plot(Nlist, np.mean(error_DNN_train_total, axis = 1), '-o', label = 'DNN + POD train')
plt.plot(Nlist, np.mean(error_DNN_train_total, axis = 1) + np.std(error_DNN_train_total, axis = 1), '--o', label = 'DNN + POD train + std' )    
# plt.plot(Nlist, np.mean(error_DNN_test, axis = 1), '-o', label = 'DNN test')
plt.plot(Nlist, np.mean(error_DNN_test_total, axis = 1), '-o', label = 'DNN + POD test')
plt.plot(Nlist, np.mean(error_DNN_test_total, axis = 1) + np.std(error_DNN_test_total, axis = 1), '--o', label = 'DNN + POD test + std' )
plt.plot(errorPOD[:Nlist[-1]], '--', label = 'Error POD')
plt.ylim(1.0e-10,1.0e-4)
plt.legend()
plt.ylabel('Mean Square Error')
plt.xlabel('N')
plt.grid()
plt.yscale('log')    


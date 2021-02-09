import os, sys
import tensorflow as tf
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Pretty print
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
         if epoch % 100 == 0: print('')
         print('.', end='')

# Network settings
L = 2 # nb of hidden layers
n = 1 # input size
m = 1 # output size
Nneurons = 10 # nb neurons hidden layers
lr = 1.0e-3 # learning rate
EPOCHS = 100 # epochs to be trained
ratio_val = 0.2 # validation ration

labelfigs = '_L2_NN10_lr1m3'

# Creation of training dataset 
Ns = 1000 # size dataset
deltaNoise = 0.05

np.random.seed(10)
X = np.random.rand(Ns)
Y = 5.0 + X*X + deltaNoise*np.random.randn(Ns)

X = X.reshape((Ns,n))
Y = Y.reshape((Ns,m))

scalerX = MinMaxScaler()
scalerX.fit(X)

scalerY = MinMaxScaler()
scalerY.fit(Y)

X_t = scalerX.transform(X)
Y_t = scalerY.transform(Y)

# Options activations: tf.nn.{tanh, sigmoid, leaky_relu, relu, linear}

# Building the architecture : L (hidden layers) + input + output layer
layers = [tf.keras.layers.Dense( Nneurons, activation=tf.keras.activations.linear, input_shape=(n,))]
for i in range(L):
    layers.append(tf.keras.layers.Dense( Nneurons, activation=tf.nn.relu))
layers.append(tf.keras.layers.Dense( m, activation=tf.nn.sigmoid )) 

model = tf.keras.Sequential(layers)

# Setting the opmisation algorithm
# Options optimisers: Adadelta, Adagrad, Adam, Adamax, FTRL, NAdam, RMSprop, SGD: 
optimizer= tf.keras.optimizers.Adam(learning_rate = lr)
model.compile(loss = ['mse'], optimizer=optimizer, metrics=['mse','mae'])

# Fitting 
hist = model.fit(X_t, Y_t, epochs=EPOCHS, validation_split=ratio_val, verbose=1, callbacks=[PrintDot() ], batch_size = 32)
model.save_weights('weights')
# model.load_weights('weights')


# Creation of test sample
Nt = 100
Xtest = np.random.rand(Nt)
Ytest = 5.0 + Xtest*Xtest + deltaNoise*np.random.randn(Nt)

Xtest = Xtest.reshape((Nt,n))
Ytest = Ytest.reshape((Nt,m))

Ypred = scalerY.inverse_transform( model.predict(scalerX.transform(Xtest)))

# Plots
plt.figure(1)
plt.plot(hist.history['mse'], label = 'train')
plt.plot(hist.history['val_mse'], label = 'validation')
plt.plot([0,99],2*[deltaNoise**2], '--',  label = 'noise square' )
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('historic_{0}.png'.format(labelfigs))
plt.show()

plt.figure(2)
plt.plot(Xtest[:,0], Ytest[:,0], 'o', label = 'test')
plt.plot(Xtest[:,0], Ypred[:,0], 'o', label = 'prediction')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('prediction_{0}.png'.format(labelfigs))
plt.show()
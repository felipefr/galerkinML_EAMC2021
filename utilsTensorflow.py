#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:37:49 2021
Availabel in: https://github.com/felipefr/galerkinML_EAMC2021.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com
"""

import tensorflow as tf
import numpy as np

from functools import partial, update_wrapper

# annotate function: Note that this avoid some errors in tensorflow
def partial2(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

# weighted mse
def custom_loss_mse(weight):
    return lambda y_p, y_d : tf.reduce_mean(tf.multiply(weight,tf.reduce_sum(tf.square(tf.subtract(y_p,y_d)),axis = 0)))

# Pretty print
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
         if epoch % 100 == 0: print('')
         print('.', end='')


# Adaptative learning
def scheduler(epoch, decay, lr, EPOCHS):    
    omega = np.sqrt(float(epoch/EPOCHS))
    rate = lr*(1.0 - omega) + omega*decay*lr
    print('learning_rate = ', rate)
    return rate
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:37:49 2021
Availabel in: https://github.com/felipefr/galerkinML_EAMC2021.git
@author: Felipe Figueredo Rocha, f.rocha.felipe@gmail.com

Wrapper Library to use h5py: Useful for to store large files
"""

import h5py
import numpy as np
from functools import partial

defaultCompression = {'dtype' : 'f8',  'compression' : "gzip", 
                           'compression_opts' : 0, 'shuffle' : False, 'chunks': (5,100)}

toChunk = lambda a: tuple([1] + [a[i] for i in range(1,len(a))])

def savehd5(filename, X,label, mode):
    with h5py.File(filename, mode) as f:
        if(type(label) == type('l')):
            defaultCompression['chunks'] = toChunk(X.shape)
            f.create_dataset(label, data = X, **defaultCompression)
        else:
            for i,l in enumerate(label):
                defaultCompression['chunks'] = toChunk(X[i].shape)
                f.create_dataset(l, data = X[i], **defaultCompression)


def loadhd5(filename, label):
    with h5py.File(filename, 'r') as f:
        if(type(label) == type('l')):
            X = np.array(f[label])
        else:
            X = []
            for l in label:
                X.append(np.array(f[l]))
    return X
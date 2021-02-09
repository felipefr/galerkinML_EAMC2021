import h5py
import numpy as np
from functools import partial
# import sys; sys.__stdout__ = sys.__stderr__ # workaround

# defaultCompression = {'dtype' : 'f8',  'compression' : "gzip", 
#                            'compression_opts' : 1, 'shuffle' : False}

defaultCompression = {'dtype' : 'f8',  'compression' : "gzip", 
                           'compression_opts' : 0, 'shuffle' : False, 'chunks': (5,100)}

toChunk = lambda a: tuple([1] + [a[i] for i in range(1,len(a))])


def merge(filenameInputs, filenameOutputs, InputLabels, OutputLabels, axis = 0, mode = 'w-'):
    

    with h5py.File(filenameOutputs, mode) as ff:
        for li,lo in zip(InputLabels,OutputLabels):  
            d = []
            for fn in filenameInputs:
                with h5py.File(fn,'r') as f:
                    if('attrs' in li):
                        s = li.split('/attrs/')
                        d.append(np.array([f[s[0]].attrs[s[1]]]))
                    else:
                        d.append(np.array(f[li])) 

            ff.create_dataset(lo, data = np.concatenate(d, axis = axis), compression = 'gzip', 
                              compression_opts = 1, shuffle = True)


def zeros_openFile(filename, shape, label, mode = 'w-'):
    f = h5py.File(filename, mode)
    
    if(type(label) == type('l')):
        defaultCompression['chunks'] = toChunk(shape)
        f.create_dataset(label, shape =  shape , **defaultCompression)
        X = f[label]
    else:
        X = []
        for i,l in enumerate(label):
            defaultCompression['chunks'] = toChunk(shape[i])
            f.create_dataset(l, shape =  shape[i] , **defaultCompression)
            X.append(f[l])
    
    return X, f

def txt2hd5(filenameIn, filenameOut, label, reshapeLast = [False,0], mode = 'w-'):
    with h5py.File(filenameOut, mode) as f:
        if(type(label) == type('l')):
            data = np.loadtxt(filenameIn)
            if(reshapeLast[0]):
                data = np.reshape(data,(data.shape[0],-1,reshapeLast[1]))
            defaultCompression['chunks'] = toChunk(data.shape)
            f.create_dataset(label, data = data, **defaultCompression)
        else:
            for i,l in enumerate(label):
                data = np.loadtxt(filenameIn[i])
                if(reshapeLast[i][0]):
                    data = np.reshape(data,(data.shape[0],-1,reshapeLast[i][1]))
                    
                defaultCompression['chunks'] = toChunk(data.shape)
                f.create_dataset(l, data = data, **defaultCompression)
       
        
def addDataset(f,X, label):
    defaultCompression['chunks'] = toChunk(X.shape)
    f.create_dataset(label, data = X , **defaultCompression)
    
def extractDataset(filenameIn, filenameOut, label, mode):
    X = loadhd5(filenameIn, label)
    savehd5(filenameOut,X,label,mode)

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

def loadhd5_openFile(filename, label, mode = 'r'):
    f = h5py.File(filename, mode)
    if(type(label) == type('l')):
        X = f[label]
    else:
        X = []
        for l in label:
            X.append(f[l])

    return X, f

def getLoadfunc(namefile, label):
       
    loadfunc = np.loadtxt
    if(namefile[-3:]=='hd5'):
        loadfunc = partial(loadhd5, label = label)
            
    return loadfunc

genericLoadfile = lambda x, y : getLoadfunc(x,y)(x)

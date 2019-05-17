#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:27:15 2019

@author: mooze
"""

import sys
if not '..' in sys.path:
    sys.path = ['..'] + sys.path

import numpy as np

import os
import pickle
import glob

from _cs231n.neural_nets import layers, models
from _cs231n.gradient_checks import check_model_gradient

## ================
## Getting data
## ================
#
#PATH_TO_DATA = '../data/cifar-10-batches-py'
#
#X_train = []
#y_train = []
#
#for fname in glob.glob(os.path.join(PATH_TO_DATA, 'data_batch_*')):
#    with open(fname, 'rb') as fh:
#        batch = pickle.load(fh, encoding='bytes')
#        
#    X_train.append(batch[b'data'])
#    y_train += batch[b'labels']
#    
#X_train = np.vstack(X_train).astype(np.float64)/255
#y_train = np.array(y_train)
#
## ================
## Splitting data
## ================
#
#num_val = 10000
#val_idxs = np.random.choice(np.arange(X_train.shape[0]), num_val, replace=False)
#train_idxs = np.array(list(set(range(X_train.shape[0])).difference(val_idxs)))
#
## ================
## Initial loss
## ================

#model = models.SequentialModel([
#    layers.DenseLayer('FC1', features_in=10, features_out=10, include_bias=True, 
#                      reg_coef=1e1), 
##    layers.PReLU('ReLU1'), 
#    layers.DenseLayer('FC2', features_in=10, features_out=10, include_bias=True, 
#                      reg_coef=1e1), 
#    layers.DenseLayer('FC3', features_in=10, features_out=10, include_bias=True, 
#                      reg_coef=1e1), 
#])

model = models.SequentialModel([
    layers.DenseLayer('FC1', 25, 20, reg_coef=1.0), 
    layers.PReLU('ReLU1'), 
    layers.DenseLayer('FC2', 20, 15, reg_coef=1.0), 
    layers.PReLU('ReLU2'), 
    layers.DenseLayer('FC3', 15, 10, reg_coef=1.0), 
    layers.PReLU('ReLU3'), 
    layers.DenseLayer('FC4', 10, 5, reg_coef=1.0), 
    layers.PReLU('ReLU4'), 
    layers.DenseLayer('FC5', 5, 5, reg_coef=1.0), 
])

X_deb = np.random.uniform(0, 1, size=(5, 25))
y_deb = np.random.choice([0, 1], size=5)
    
check_model_gradient(model, X_deb, rel_delta_X=1e-3, reltol=1e-6, 
                     out_fct=lambda X: models.softmax_predict(X, y_deb)[1:3])
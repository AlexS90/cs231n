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
from _cs231n.metrics import accuracy

# ================
# Getting data
# ================

PATH_TO_DATA = '../data/cifar-10-batches-py'

X_train0 = []
y_train0 = []

for fname in glob.glob(os.path.join(PATH_TO_DATA, 'data_batch_*')):
    with open(fname, 'rb') as fh:
        batch = pickle.load(fh, encoding='bytes')
        
    X_train0.append(batch[b'data'])
    y_train0 += batch[b'labels']
    
X_train0 = np.vstack(X_train0).astype(np.float64)/255
y_train0 = np.array(y_train0)

# ================
# Splitting data
# ================

num_val = 10000
val_idxs = np.random.choice(np.arange(X_train0.shape[0]), num_val, replace=False)
train_idxs = np.array(list(set(range(X_train0.shape[0])).difference(val_idxs)))

X_mean = X_train0[train_idxs].mean(axis=0)
X_std = X_train0[train_idxs].std(axis=0)

X_train_norm = (X_train0[train_idxs[:10000]] - X_mean[None, :])/X_std
y_train = y_train0[train_idxs[:10000]]
X_val_norm = (X_train0[val_idxs[:2000]] - X_mean[None, :])/X_std
y_val = y_train0[val_idxs[:2000]]

# ================
# Model
# ================

#model = models.SequentialModel([
#    layers.DenseLayer('FC1', features_in=3072, features_out=256, include_bias=True, 
#                      reg_coef=1e-4), 
#    layers.BatchNorm('BN1'), 
#    layers.Dropout('Drop_FC1', p_drop=0.35), 
#    layers.PReLU('ReLU1'), 
#    layers.DenseLayer('FC2', features_in=256, features_out=64, include_bias=True, 
#                      reg_coef=1e-4), 
#    layers.BatchNorm('BN2'), 
#    layers.Dropout('Drop_FC2', p_drop=0.35), 
#    layers.PReLU('ReLU2'), 
#    layers.DenseLayer('FC3', features_in=64, features_out=10, include_bias=True, 
#                      reg_coef=1e-4)
#])

X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], 
                                    3, 32, 32).transpose((0, 2, 3, 1))
X_val_norm = X_val_norm.reshape(X_val_norm.shape[0], 
                                3, 32, 32).transpose((0, 2, 3, 1))

model = models.SequentialModel([
    #Shape: 32 x 32 x 3
    layers.Conv2D('Conv1', channels_in=3, channels_out=8, 
                  filter_size=5, stride=1, padding=0, 
                  reg_coef=1e-3), 
    #Shape: 28 x 28 x 8
    layers.MaxPooling2D('MaxPool1', pool_size=2, stride=2), 
    #Shape: 14 x 14 x 8
    layers.Resizer('Flattener', shape_in=(14, 14, 8), shape_out=(1568, )), 
    #Shape: 1568
    layers.DenseLayer('FC1', features_in=1568, features_out=256, 
                      reg_coef=1e-3), 
    #Shape: 256
    layers.ReLU('ReLU1'), 
    layers.DenseLayer('FC2', features_in=256, features_out=10, 
                      reg_coef=1e-3)
    #Shape: 10
])


pred_train, _, loss_train, _ = model.predict(X_train_norm, y_train, batch_size=128)
pred_val, _, loss_val, _ = model.predict(X_val_norm, y_val, batch_size=128)

acc_train = np.mean(pred_train == y_train)
acc_val = np.mean(pred_val == y_val)

print(f'Initial training loss: {loss_train}')
print(f'Initial validation loss: {loss_val}')
print(f'Initial training accuracy: {acc_train}')
print(f'Initial validation accuracy: {acc_val}')

# ----------------

loss_train_hist, loss_val_hist, acc_val_hist = \
models.train_SGD(model, models.softmax_predict, 
                 X_train_norm, y_train, 
                 X_val_norm, y_val, 
                 epochs=5, batch_size=128, 
                 metric=accuracy, 
                 lr=1e-2, lr_decrease_step=1, lr_decrease_coef=0.95, momentum=0.0, 
                 show_progress=True)

# ----------------

#model_out_train = model.forward(X_train_norm, train_pass=False)
#model_out_val = model.forward(X_val_norm, train_pass=False)
#
#pred_train, loss_train = models.softmax_predict(model_out_train, y_train)[:2]
#pred_val, loss_val = models.softmax_predict(model_out_val, y_val)[:2]
#
#acc_train = np.mean(pred_train == y_train)
#acc_val = np.mean(pred_val == y_val)
#
#print(f'Final training loss: {loss_train}')
#print(f'Final validation loss: {loss_val}')
#print(f'Final training accuracy: {acc_train}')
#print(f'Final validation accuracy: {acc_val}')
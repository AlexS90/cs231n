#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:25:13 2019

@author: mooze
"""

import numpy as np
from collections import OrderedDict

# ================
# Implementing my own models
# ===============

class Model():
    def __init__(self, layers):
        self.layers = OrderedDict((xlayer.name, xlayer) for xlayer in layers)
        
        for xlayer in self.layers.values():
            for xparam in xlayer.params():
                xparam.name = '.'.join([xlayer.name, xparam.name])
    
    def forward(self, X, train_pass=True):
        pass
    
    def backward(self, grad_Y):
        pass
    
    def params(self):
        model_params = []
        
        for xlayer in self.layers.values():
            model_params += xlayer.params()
        
        return model_params
    
    def reg_loss(self):
        loss = 0.0
        
        for xlayer in self.layers.values():
            loss += xlayer.reg_loss()
            
        return loss
    
    def reset_grad(self):
        for xlayer in self.layers.values():
            xlayer.reset_grad()

# ===============
    
class SequentialModel(Model):
    def __init__(self, layers):
        super(SequentialModel, self).__init__(layers)
        
    def forward(self, X, train_pass=True):
        Y = X.copy()
        
        for xlayer in self.layers.values():
            if train_pass:
                Y = xlayer.forward(Y, True)
            else:
                if xlayer.required_on_inference:
                    Y = xlayer.forward(Y, False)
                
        return Y
    
    def backward(self, grad_Y):
        grad_X = grad_Y.copy()
        
        for xlayer in reversed(self.layers.values()):
            grad_X = xlayer.backward(grad_X)
            
        return grad_X
    
# ================
        
def softmax_predict(X, y=None):
    """
    Performs softmax classification on X.
    If y is not None, returns loss and gradient as well
    
    Returns:
        (pred, loss, grad, probs)
    """
    
    scores = np.exp(X - X.max(axis=1)[:, None])
    probs = scores/(scores.sum(axis=1)[:, None])
    
    if y is None:
        return probs.argmax(axis=1), None, None, probs
    else:
        loss = -np.log(probs[np.arange(X.shape[0]), y]).sum()/X.shape[0]
        grad = probs
        grad[np.arange(X.shape[0]), y] -= 1.0
        
        return probs.argmax(axis=1), loss, grad/X.shape[0], probs
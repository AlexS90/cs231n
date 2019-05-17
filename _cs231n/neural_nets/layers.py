#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:36:05 2019

@author: mooze
"""

# ================
# Implementing my own layers
# ===============

import numpy as np
        
# ================
# ================
# ================

def reg_L2(reg_coef, par_value):
    reg_loss = reg_coef*np.sum(par_value**2)
    reg_grad = 2*reg_coef*par_value
    return (reg_loss, reg_grad)

def reg_L1(reg_coef, par_value):
    reg_loss = reg_coef*np.sum(np.abs(par_value))
    
    neg_mask = par_value < 0
    reg_grad = np.full_like(par_value, reg_coef)
    reg_grad[neg_mask] = -reg_grad[neg_mask]
    return (reg_loss, reg_grad)


class Parameter:
    def __init__(self, name, init_value=None, reg_coef=0.0, reg_type=None):
        self.name = name
        
        if init_value is None:
            self.value = np.zeros((1, ), dtype=np.float64)
        else:
            self.value = init_value
            
        self.grad = np.zeros_like(self.value)
        self.reg_coef = reg_coef
        
        if reg_type == 'L2':
            self.reg_fct = reg_L2
        elif reg_type == 'L1':
            self.reg_fct = reg_L1
        else:
            self.reg_fct = lambda reg_coef, par_value: (0.0, 0.0)
            
    
    def reg_loss(self):
        return self.reg_fct(self.reg_coef, self.value)
        
        
# ================
# ================
# ================


class Layer:
    def __init__(self, name, required_on_inference=True):
        self.name = name
        self.required_on_inference=required_on_inference
    
    def forward(self, accum_grad=True):
        return None
    
    def backward(self):
        return None
    
    def params(self):
        return []
    
    def reg_loss(self):
        loss = 0.0
        
        for xparam in self.params():
            loss += xparam.reg_loss()[0]
            
        return loss
    
    def reset_grad(self):
        return None
    
# ================
# ================
# ================
        
class DenseLayer(Layer):
    def __init__(self, name, features_in, features_out, include_bias=True, 
                 reg_coef = 0.0, init_std=None, required_on_inference=True):        
        super().__init__(name, required_on_inference)
        self.features_in = features_in
        self.features_out = features_out
        
        # ================
        
        if init_std is None:
            W0 = np.random.normal(0.0, 1/np.sqrt(features_in), (features_in, features_out))
        else:
            W0 = np.random.normal(0.0, init_std, (features_in, features_out))
            
        self.W = Parameter('W', W0, reg_coef, 'L2')
            
        if include_bias:
            self.B = Parameter('B', np.full((features_out, ), 1/features_out, dtype=np.float64))
            
        self.X_fw = None
            
    
    def forward(self, X, accum_grad=True):
        if accum_grad:
            self.X_fw = X
            
        Y = np.dot(X, self.W.value)
        
        if hasattr(self, 'B'):
            Y += self.B.value[None, :]
            
        return Y
    
    
    def backward(self, grad_Y):
        if self.X_fw is None:
            raise ValueError('No arguments recorded over the last forward pass or they have been erased')
        
        grad_X = np.dot(grad_Y, self.W.value.T)
        self.W.grad = np.dot(self.X_fw.T, grad_Y) + self.W.reg_loss()[1]
        
        if hasattr(self, 'B'):
            self.B.grad = grad_Y.sum(axis=0)
            
        return grad_X
    
    
    def params(self):
        if hasattr(self, 'B'):
            return [self.W, self.B]
        else:
            return [self.W]
        
        
    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
        self.X_fw = None

      
# ================

         
class ReLU(Layer):
    def __init__(self, name, required_on_inference=True):
        super().__init__(name, required_on_inference)
        self.neg_mask = None
        
        
    def forward(self, X, accum_grad=True):
        neg_mask = X < 0.0
        
        if accum_grad:
            self.neg_mask = neg_mask
            
        Y = X.copy()
        Y[neg_mask] = 0.0
        
        return Y
    
    
    def backward(self, grad_Y):
        if self.neg_mask is None:
            raise ValueError('No arguments recorded over the last forward pass or they have been erased')
        
        grad_X = grad_Y.copy()
        grad_X[self.neg_mask] = 0.0
        
        return grad_X
    
    
    def reset_grad(self):
        self.neg_mask = None

# ================
    
    
class PReLU(Layer):
    def __init__(self, name, neg_slope=None, required_on_inference=True):
        super().__init__(name, required_on_inference)
        
        if neg_slope is None:
            neg_slope0 = np.array([-0.01])
        
        self.neg_slope = Parameter('negative_slope', neg_slope0)
        self.neg_mask = None
        self.X_fw = None
        
        
    def forward(self, X, accum_grad=True):
        neg_mask = X < 0.0
        
        if accum_grad:
            self.neg_mask = neg_mask
            self.X_fw = X
            
        Y = X.copy()
        Y[neg_mask] *= self.neg_slope.value[0]
        
        return Y
    
    
    def backward(self, grad_Y):
        if self.neg_mask is None:
            raise ValueError('No arguments recorded over the last forward pass or they have been erased')
        
        self.neg_slope.grad = np.array([
            np.sum(grad_Y[self.neg_mask]*self.X_fw[self.neg_mask])
        ])
        
        grad_X = grad_Y.copy()
        grad_X[self.neg_mask] *= self.neg_slope.value[0]
        
        return grad_X
    
    
    def params(self):
        return [self.neg_slope]
    
    
    def reset_grad(self):
        self.neg_slope.grad = np.zeros_like(self.neg_slope.value)
        self.neg_mask = None
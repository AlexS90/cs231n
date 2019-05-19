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
        return(sum(xparam.reg_loss()[0] for xparam in self.params()))
            
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
            init_std = np.sqrt(2.0/features_in)
            
        self.W = Parameter('W', init_std*np.random.normal(size=(features_in, features_out)), 
                           reg_coef, 'L2')
            
        if include_bias:
            self.B = Parameter('B', np.zeros((features_out, ), dtype=np.float64))
            
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
        
# ================
# ================
# ================
        
class Dropout(Layer):
    def __init__(self, name, p_drop=0.5, required_on_inference=False):
        super().__init__(name, required_on_inference)
        self.p_drop = p_drop
        self.dropped_mask = None
        
        
    def forward(self, X, accum_grad=True):
        if accum_grad:
            self.dropped_mask = np.random.choice([False, True], X.shape, 
                                                 p=(1 - self.p_drop, self.p_drop))
        
        Y = X.copy()
        Y[self.dropped_mask] = 0.0
        
        return Y/(1 - self.p_drop)
    
    
    def backward(self, grad_Y):
        grad_X = grad_Y.copy()
        grad_X[self.dropped_mask] = 0.0
        
        return grad_X/(1 - self.p_drop)
    
    
    def reset_grad(self):
        self.dropped_mask = None


class BatchNorm(Layer):
    def __init__(self, name, required_on_inference=True, eps=1e-6):        
        super().__init__(name, required_on_inference)
        self.eps = eps
        self.scale = Parameter('scale', np.array([1.0]))
        self.offset = Parameter('offset', np.array([0.1]))
        
        self.X_cent_fw = None
        self.batch_var_fw = None
        
        
    def forward(self, X, accum_grad=True):
        batch_mean = X.mean()
        batch_var = X.var()
        X_cent = (X - batch_mean)/np.sqrt(batch_var + self.eps)
        
        if accum_grad:
            self.X_cent_fw = X_cent
            self.batch_var_fw = batch_var
        
        return self.scale.value*X_cent + self.offset.value
        
    def backward(self, grad_Y):
        self.scale.grad = np.array([np.sum(grad_Y*self.X_cent_fw)])
        self.offset.grad = np.array([np.sum(grad_Y)])
        
        
        grad_X = grad_Y - grad_Y.sum()/grad_Y.size - \
            self.X_cent_fw*np.sum(grad_Y*self.X_cent_fw)/grad_Y.size
            
        return grad_X/np.sqrt(self.batch_var_fw + self.eps)
    
    
    def params(self):
        return [self.scale, self.offset]
    
    
    def reset_grad(self):
        self.scale.grad = 0.0
        self.offset.grad = 0.0
        
        self.X_cent_fw = None
        self.batch_var_fw = None
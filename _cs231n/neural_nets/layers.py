#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:36:05 2019

@author: mooze
"""


import numpy as np
from scipy.sparse import csr_matrix
import itertools as it


# ================================================================
# ================================================================
# ================================================================


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


# ================================================================
# ================================================================
# ================================================================


class Layer:
    def __init__(self, name, required_on_inference=True):
        self.name = name
        self.required_on_inference=required_on_inference
    
    def forward(self, X, accum_grad=True):
        return None
    
    def backward(self, grad_Y):
        return None
    
    def params(self):
        return []
    
    def reg_loss(self):
        return(sum(xparam.reg_loss()[0] for xparam in self.params()))
    
    def reset_grad(self):
        return None


# ================================================================
# ================================================================
# ================================================================


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
            self.B = Parameter('B', 0.1*np.ones((features_out, ), dtype=np.float64))
            
        self.X_fw = None
            
    
    def forward(self, X, accum_grad=True):
        if accum_grad:
            self.X_fw = X.copy()
            
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
        self.X_fw = None
        self.W.grad = np.zeros_like(self.W.value)
        
        if hasattr(self, 'B'):
            self.B.grad = np.zeros_like(self.B.value)


# ================================================================
# ================================================================
# ================================================================


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


# ================================================================
# ================================================================
# ================================================================


class Conv2D(Layer):
    def __init__(self, name, channels_in, channels_out=1, 
                 filter_size=3, stride=1, padding=0, 
                 include_bias=True, reg_coef=0.0, required_on_inference=True):
        super().__init__(name, required_on_inference)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        self.W = Parameter('W', 1/np.sqrt(filter_size**2*channels_in*channels_out)*\
            np.random.normal(size=(filter_size, filter_size, channels_in, channels_out)), 
            reg_coef=reg_coef, reg_type='L2')
        
        if include_bias:
            self.B = Parameter('B', 0.1*np.ones((channels_out, ), dtype=np.float64))
        
        # --------------------------------
        
        self.xshape = None
        
        tmp = np.array(list(it.product(
            range(self.filter_size), 
            range(self.filter_size), 
            range(self.channels_in)
        )))
        
        self.step_Xh = tmp[:, 0]
        self.step_Xw = tmp[:, 1]
        
        self.Xh = None
        self.Xw = None
        self.Xc = np.array(list(range(channels_in))*filter_size**2, dtype=np.uint16)
        self.Yh = None
        self.Yw = None
        
    
    def pad_X(self, X):
        if self.padding == 0:
            return X
        else:
            return np.concatenate([
                np.zeros((X.shape[0], self.padding, X.shape[2] + 2*self.padding, X.shape[3])), 
                np.concatenate([
                    np.zeros((X.shape[0], X.shape[1], self.padding, X.shape[3])),
                    X, 
                    np.zeros((X.shape[0], X.shape[1], self.padding, X.shape[3]))
                ], axis=2), 
                np.zeros((X.shape[0], self.padding, X.shape[2] + 2*self.padding, X.shape[3]))
            ], axis=1)
        
        
    def ravel_X(self, X):
        return X[:, self.Xh.ravel(), self.Xw.ravel(), np.tile(self.Xc, len(self.Xh))].\
                   reshape(X.shape[0], -1, self.filter_size**2*self.channels_in)
        
        
    def forward(self, X, accum_grad=True):
        if (self.xshape is None) or (self.xshape != X.shape[1:3]):
            assert (X.shape[1] + 2*self.padding - self.filter_size)%self.stride == 0
            assert (X.shape[2] + 2*self.padding - self.filter_size)%self.stride == 0
            assert X.shape[3] == self.channels_in
            self.xshape = X.shape[1:3]
            
            tmp = np.array(list(it.product(
                range((X.shape[1] + 2*self.padding - self.filter_size)//self.stride + 1), 
                range((X.shape[2] + 2*self.padding - self.filter_size)//self.stride + 1)
            )))
            
            self.Yh = tmp[:, 0].astype(np.uint16)
            self.Yw = tmp[:, 1].astype(np.uint16)

            self.Xh = np.array([self.stride*Yh + self.step_Xh for Yh in self.Yh]).astype(np.uint16)
            self.Xw = np.array([self.stride*Yw + self.step_Xw for Yw in self.Yw]).astype(np.uint16)
            
        # --------------------------------
        
        if accum_grad:
            self.fw_X = self.pad_X(X)
        
        Y = np.einsum('bij, jk -> bik', 
                      self.ravel_X(self.pad_X(X)), 
                      self.W.value.reshape(-1, self.channels_out))
        
        if hasattr(self, 'B'):
            Y += self.B.value[None, None, :]
            
        return Y.reshape(X.shape[0], self.Yh[-1] + 1, self.Yw[-1] + 1, self.channels_out)
    
    
    def backward(self, grad_Y):
        assert grad_Y.shape[1:] == (self.Yh[-1] + 1, self.Yw[-1] + 1, self.channels_out)
        
        grad_Y_raveled = grad_Y.reshape(grad_Y.shape[0], -1, self.channels_out)
        grad_X_raveled = np.einsum('bik, jk -> bij', 
                                   grad_Y_raveled, 
                                   self.W.value.reshape(-1, self.channels_out))
        grad_X = np.zeros((*self.fw_X.shape, *grad_X_raveled.shape[1:]), dtype=np.float64)
        
        grids = np.mgrid[:grad_X_raveled.shape[1], :grad_X_raveled.shape[2]]
        grad_X[:, self.Xh.ravel(), self.Xw.ravel(), 
               np.tile(self.Xc, self.Xh.shape[0]), 
               grids[0].ravel(), grids[1].ravel()] = \
               grad_X_raveled.reshape(grad_Y.shape[0], -1)
        grad_X = grad_X.sum(axis=(4, 5))
        
        self.W.grad = np.einsum('bkj, bki -> ij', 
                                grad_Y_raveled, 
                                self.ravel_X(self.fw_X)).\
            reshape(self.filter_size, self.filter_size, self.channels_in, self.channels_out) + \
            self.W.reg_loss()[1]
            
        if hasattr(self, 'B'):
            self.B.grad = np.einsum('bki -> i', grad_Y_raveled)
        
        if self.padding > 0:
            return grad_X[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            return grad_X
    
    
    def params(self):
        if hasattr(self, 'B'):
            return [self.W, self.B]
        else:
            return [self.W]
        
        
    def reset_grad(self):
        self.fw_X = None
        self.W.grad = np.zeros_like(self.W.value)
        
        if hasattr(self, 'B'):
            self.B.grad = np.zeros_like(self.B.value)


# ================================================================
# ================================================================
# ================================================================


class Pooling2D(Layer):
    def __init__(self, name, pool_size=2, stride=2, required_on_inference=True):
        super().__init__(name, required_on_inference)
        self.pool_size = pool_size
        self.stride = stride
        
        # --------------------------------
        
        self.xshape = None
        
        tmp = np.array(list(it.product(range(self.pool_size), range(self.pool_size))))
        self.step_Xh = tmp[:, 0]
        self.step_Xw = tmp[:, 1]
        
        self.Yh = None
        self.Yw = None
        self.Xh = None
        self.Xw = None
    
    
    def forward(self, X, accum_grad=True):
        if (self.xshape is None) or (self.xshape != X.shape[1:3]):
            assert (X.shape[1] - self.pool_size)%self.stride == 0
            assert (X.shape[2] - self.pool_size)%self.stride == 0
            
            self.xshape = X.shape[1:3]
            
            tmp = np.array(list(it.product(
                range((self.xshape[0] - self.pool_size)//self.stride + 1), 
                range((self.xshape[1] - self.pool_size)//self.stride + 1)
            )))
            
            self.Yh = tmp[:, 0].astype(np.uint16)
            self.Yw = tmp[:, 1].astype(np.uint16)
            
            self.Xh = np.array([self.stride*Yh + self.step_Xh for Yh in self.Yh]).astype(np.uint16)
            self.Xw = np.array([self.stride*Yw + self.step_Xw for Yw in self.Yw]).astype(np.uint16)
            
        return None
    
    
    def ravel_X(self, X):
        return X[:, self.Xh.ravel(), self.Xw.ravel(), :].\
                   reshape((X.shape[0], self.Xh.shape[0], self.Xh.shape[1], X.shape[3]))


# ----------------------------------------------------------------


class AveragePooling2D(Pooling2D):
    def __init__(self, name, pool_size=2, stride=2, required_on_inference=True):
        super().__init__(name, pool_size, stride, required_on_inference)
    
    
    def forward(self, X, accum_grad=True):
        super().forward(X, accum_grad)
        
        Y = self.ravel_X(X).mean(axis=2).\
            reshape((X.shape[0], self.Yh[-1] + 1, self.Yw[-1] + 1, X.shape[3]))
        
        return Y
    
    
    def backward(self, grad_Y):
        assert grad_Y.shape[1:3] == (self.Yh[-1] + 1, self.Yw[-1] + 1)
        
        grad_X_raveled = np.tile(
            grad_Y.transpose((0, 3, 1, 2)).\
            reshape(grad_Y.shape[0], -1, 1, grad_Y.shape[3]), 
            (1, 1, self.Xh.shape[1], 1))
        
        grad_X = np.zeros((grad_Y.shape[0], grad_Y.shape[3], *self.xshape, 
                          *grad_X_raveled.shape[1:3]), dtype=np.float64)
        grids = np.mgrid[:grad_X_raveled.shape[1], :grad_X_raveled.shape[2]]
        grad_X[:, :, self.Xh.ravel(), self.Xw.ravel(), 
               grids[0].ravel(), grids[1].ravel()] = \
               grad_X_raveled.reshape(grad_Y.shape[0], grad_Y.shape[3], -1)
        grad_X = grad_X.sum(axis=(-2, -1))
            
        return grad_X.transpose((0, 2, 3, 1))/self.pool_size**2


# ----------------------------------------------------------------


class MaxPooling2D(Pooling2D):
    def __init__(self, name, pool_size=2, stride=2, required_on_inference=True):
        super().__init__(name, pool_size, stride, required_on_inference)
        self.Yidx = None
        
    
    def forward(self, X, accum_grad=True):
        super().forward(X, accum_grad)
        
        X_raveled = self.ravel_X(X)
        self.Yidx = X_raveled.argmax(axis=2)
        grids = np.mgrid[:X.shape[0], :X_raveled.shape[1], :X.shape[3]]
        Y = X_raveled[grids[0].ravel(), 
                      grids[1].ravel(), 
                      self.Yidx.ravel(), 
                      grids[2].ravel()].\
            reshape(X.shape[0], self.Yh[-1] + 1, self.Yw[-1] + 1, X.shape[3])
        
        return Y
    
    
    def backward(self, grad_Y):
        assert grad_Y.shape[1:3] == (self.Yh[-1] + 1, self.Yw[-1] + 1)
        
        grid_Xwh = np.hstack([[ir]*self.Yidx.shape[2] for ir in range(self.Yidx.shape[1])]*grad_Y.shape[0])
        
        grid_rows = self.Xh[grid_Xwh, self.Yidx.ravel()]
        grid_cols = self.Xw[grid_Xwh, self.Yidx.ravel()]
        grid_batch, _, grid_channel = np.mgrid[:grad_Y.shape[0], :self.Yidx.shape[1], :grad_Y.shape[3]]
        
        grad_X = csr_matrix((
            grad_Y.ravel(), 
            (
                grid_batch.ravel(), 
                np.ravel_multi_index((grid_rows, grid_cols, grid_channel.ravel()), 
                                     (*self.xshape, grad_Y.shape[3]))
            )), shape=(grad_Y.shape[0], np.prod(self.xshape)*grad_Y.shape[3])
        )
        
        return np.asarray(grad_X.todense()).\
            reshape(grad_Y.shape[0], *self.xshape, grad_Y.shape[3])
    
    
    def reset_grad(self):
        self.X_idxmax = None

# ----------------------------------------------------------------

# ================================================================
# ================================================================
# ================================================================


class Resizer(Layer):
    def __init__(self, name, shape_in, shape_out=None, required_on_inference=True):
        super().__init__(name, required_on_inference)
        
        assert np.prod(shape_in) == np.prod(shape_out)
        self.shape_in = shape_in
        self.shape_out = shape_out
        
        
    def forward(self, X, accum_grad=True):
        assert X.shape[1:] == self.shape_in
        return X.reshape((X.shape[0], *self.shape_out))
    
    
    def backward(self, grad_Y):
        assert grad_Y.shape[1:] == self.shape_out
        return grad_Y.reshape((grad_Y.shape[0], *self.shape_in))


# ================================================================
# ================================================================
# ================================================================


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


# ----------------------------------------------------------------


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
        
# ================

md = MaxPooling2D('AVG1', pool_size=3, stride=2)

X0 = np.round(np.random.uniform(0, 1, (4, 5, 7, 3)), 3)
Y0 = md.forward(X0)

grad_Y0 = np.round(np.random.uniform(0, 1, Y0.shape), 3)
#grad_Y0[:, :, :, 1] += 1
#grad_Y0[:, :, :, 2] += 2
grad_X0 = md.backward(grad_Y0)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:25:13 2019

@author: mooze
"""

import numpy as np
from tqdm import tqdm
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
# Implementing my own models
# ===============
        
def softmax_predict(X, y=None):
    """
    Performs softmax classification on X.
    If y is not None, returns loss and gradient as well
    
    Returns:
        (pred, loss, grad, probs)
    """
    
    scores = np.exp(X - X.max(axis=1)[:, None])
    probs = scores/(scores.sum(axis=1)[:, None])
    hard_out = np.argmax(probs, axis=1)
    
    if y is None:
        return hard_out, None, None, probs
    else:
        loss = -np.log(probs[np.arange(X.shape[0]), y]).sum()/X.shape[0]
        grad = probs.copy()
        grad[np.arange(X.shape[0]), y] -= 1.0
        
        return hard_out, loss, grad/X.shape[0], probs
    
# ================
# Implementing my own models
# ===============
        
def train_SGD(model, fct_loss, 
               X_train, y_train, X_val=None, y_val=None, shuffle=True, 
               epochs=5, batch_size=64, metric=None, parameters=None, 
               lr=1e-3, lr_decrease_step=5, lr_decrease_coef=0.5, momentum=0.1, 
               show_progress=True, verbose=True):
    """
    Performs one or more epochs of model training on X_train, y_train in batches, 
    using SGD with momentum and validating results on X_val, y_val. 
    It is possible to train only selected 
    elements of model, passing them to layers input variable.
    
    
    Parameters:
    ----------------
    model : instance of Model subclass
        model to be trained
    fct_loss : function np.ndarray, np.ndarray -> (float, np.ndarray)
        loss function that takes model output and returns a tuple with
        following:
            0 - np.ndarray of hard model predictions
            1 - float, loss value
            2 - np.ndarray, gradient of loss with respect to model output
            3 - np.ndarray of soft model predictions
    X_train : np.ndarray
        training data. First dimension is number of instances
    y_train : np.ndarray
        labels of training data
    X_val : np.ndarray
        validation data. If None - no validation is performed
    y_val : np.ndarray
        validation data labels. If None - no validation is performed
    shuffle : bool
        whether to shuffle training data before passing through it or not
    epochs : int
        number of epochs to train with this setup
    batch_size : int >= 0
        length of training batch. If 0 - epoch is trained in one big batch
    metric : function np.ndarray, np.ndarray -> float
        metric to be calculated after epoch on both training and validation sets.
        Accepts ground truth as first argument and hard model predictions as second
    !!! TODO !!!
    lr : float > 0
         learning rate for gradient descend
    momentum : 1 > float > 0
        momentum coefficient for SGD descend
        new_grad = momentum*old_grad + (1 - momentum)*loss_grad
    show_progress : bool
        If True - a progress bar with batch progression will be displayed
        
    Returns:
    ----------------
    loss_history : list of float
    metric_history : list of float
    """
    
    # Initialize loss & metric histories
    loss_history_train = []
    loss_history_val = []
    metric_history = []
        
    # Prepare list of parameters to train and their rolling gradients
    if parameters is None:
        parameters = model.params()
    
    param_grad = [np.zeros_like(xparam.value) for xparam in parameters]
    
    print('\n' + '-'*32 + '\n')
    
    #Main cycle over epochs
    for xepoch in range(epochs):
        if verbose:
            print(f'Epoch {xepoch + 1}/{epochs}...')
            print(f'Learning rate: {lr}')
            
        # Shuffle the training data indices if required
        train_idxs = np.arange(X_train.shape[0])
        if shuffle:
            train_idxs = np.random.permutation(train_idxs)
            
        # Splitting indices into batches
        if batch_size == 0:
            batch_idxs = [train_idxs]
        else:
            batch_idxs = np.array_split(train_idxs, max(len(train_idxs)//batch_size, 1))
            
        if show_progress:
            pbar = tqdm(total=len(batch_idxs))
        
        # --------------------------------
        # Cycle over batches. Necessary to reset gradients beforehands
        batch_loss = []
        model.reset_grad()
        
        for xbatch_idxs in batch_idxs:
            # Forward and loss gradient
            model_out = model.forward(X_train[xbatch_idxs, :], train_pass=True)
            loss, loss_grad = fct_loss(model_out, y_train[xbatch_idxs])[1:3]
            batch_loss.append(loss)
            
            # Backward pass
            model.backward(loss_grad)
            
            # Recalculate gradient and make SGD step
            for (xparam, xgrad) in zip(parameters, param_grad):
                xgrad = momentum*xgrad + (1 - momentum)*xparam.grad
                xparam.value -= lr*xgrad
                
            # Zero gradients
            model.reset_grad()
                
            if show_progress:
                pbar.update()
                
        # --------------------------------
        # Passed over batches - now accumulate epoch loss as average over batches
        
        if show_progress:
            pbar.close()
        
        loss_history_train.append(np.mean(batch_loss))
        
        # Is there is a validation set - calculate loss on it
        if not X_val is None:
            model_out = model.forward(X_val, train_pass=False)
            model_pred, val_loss = fct_loss(model_out, y_val)[:2]
            loss_history_val.append(val_loss)
            
            # If there is a metric - calculate
            if metric:
                metric_history.append(metric(y_val, model_pred))
                
        # Output to stdin
        if verbose:
            print(f'Epoch is over. Training loss: {loss_history_train[-1]}')
            
            if not X_val is None:
                print(f'Validation loss: {loss_history_val[-1]}')
                
                if metric:
                    print(f'Validation metric: {metric_history[-1]}')
                    
            print('\n' + '-'*32 + '\n')
            
        # --------------------------------
        # Decreasing learning rate
        
        if (xepoch + 1)%lr_decrease_step == 0:
            lr *= lr_decrease_coef
            
    # Main cycle over
    
    return loss_history_train, loss_history_val, metric_history
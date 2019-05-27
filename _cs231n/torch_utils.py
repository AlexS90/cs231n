#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:49:19 2019

@author: mooze
"""

import torch
from torch import nn

from tqdm import tqdm_notebook

import numpy as np
import matplotlib.pyplot as plt

import time


def predict_torch(model, dataloader, device, 
                  loss_fct=torch.nn.functional.cross_entropy, 
                  out_fct=torch.nn.functional.softmax):
    """
    Runs entire dataloader contents through model on given device and
    returns predictions along with calculated loss.
    
    Parameters
    ----------------
    model : torch.nn.Module subclass object
        Model for inference
    dataloader : torch.data.utils.DataLoader object
        Data loader for a dataset. Must support batch sampling
    device : 
        Device (either CPU or GPU to perform inference on)
    loss_fct : function
        Takes raw model output and labels and returns loss value.
        Default is softmax coupled with cross-entropy
    out_fct : function
        Takes raw model output and returns normalized probabilities
        
    Returns
    ----------------
    resX : torch.tensor
        Normalized model output
    resY : torch.tensor
        Output labels
    loss : loss value over dataset
    
    Notes
    ----------------
    Model predictions are not stored on device, but in RAM, and transformed
    into NumPy arrays.
    First dimension of resX corresponds to batch indexing.
    """
    
    # Initialize model to inference and prepare containers for output
    model.eval()
    resX = []
    resy = []
    
    loss = 0.0
    lenx = 0
    
    # Cycle through batches of data
    for (X, y) in dataloader:
        # Move data to device and get prediction
        X_dev = X.to(device)
        y_dev = y.to(device)
        pred = model(X_dev)
        
        # Standard torch loss is averaged over batch size. Original value is 
        # scaled back.
        lenx += len(X)
        loss += loss_fct(pred, y_dev).item()*len(X)
        
        # Transport model predictions from device to RAM
        resX.append(out_fct(pred, dim=1).detach().cpu())
        resy.append(y)
        
    # Concatenating batch predictions
    resX = torch.cat(resX, dim=0)
    resy = torch.cat(resy, dim=0)
    
    # Also average loss over all dataset objects
    return resX, resy, loss/lenx


def initial_model_check(model, train_loader, val_loader, device, metric):
    """
    Performs sanity check whether a model has initialized correctly and also
    calculated initial metric value.
    
    Function calculates loss function on both training and validation set, as
    well as validation metric value.
    
    Parameters
    ----------------
    model : torch.nn.Module subclass instance
        Model to be checked
    train_loader, val_loader : torch.utils.data.DataLoader
        Data loader for train/validation sets
    device : 
        Device (either CPU or GPU to perform inference on)
    metric : function -> float
        Takes true labels and model predictions and returns desired metric value
    """
    
    # Calculate loss on train dataset
    loss_tr = predict_torch(model, train_loader, device, 
                            loss_fct=nn.CrossEntropyLoss())[2]
    print('Model loss on train set: {0:.3f}'.format(loss_tr))
    
    # Calculate model predictions, loss and metric on validation dataset
    predX, gt, loss_val = predict_torch(model, val_loader, device, 
                                        loss_fct=nn.CrossEntropyLoss())
    metric_val = metric(gt.numpy(), predX.numpy().argmax(axis=1))
    print('Model loss on validation set: {0:.3f}'.format(loss_val))
    print('Model metric on validation set set: {0:.5f}'.format(metric_val))
    
    return loss_tr, loss_val, metric_val


def train_torch(model, device, train_loader, val_loader, 
                loss_fct, opt, lr_sch, metric=None, 
                num_epochs=10, 
                verbose=True, show_progress=True, delay=0):
    """
    Wrapper around basic torch training process. Performs optimization
    of model parameters in order to minimize loss value on given dataset
    
    Parameters
    ----------------
    model : torch.nn.Module subclass instance
        Model to be checked
    device : 
        Device (either CPU or GPU to perform inference on)
    train_loader, val_loader : torch.utils.data.DataLoader
        Data loader for train/validation sets
    loss_fct : function
        Takes raw model output and labels and returns loss value.
    opt : torch.optim.Optimizer subclass instance
        Optimizer performing parameters update
    lr_sch : torch.optim.lr_scheduler.LR_scheduler subclass instance
        Learning rate scheduler
    metric : function -> float
        Takes true labels and model predictions and returns desired metric value
    num_epochs : int > 0
        Number of full passes over training data
    verbose : bool
        If True - after every batch info about losses and metrics is displayed
    show_progress : bool
        If True - tqdm progress bar is displayed for every epoch
    delay : int > 0
        If non-zero - number of seconds to wait before starting new epoch.
        Used for throttling, in case device is overheating.
        
    Returns:
    ----------------
    tr_loss : [int]
        Sequence of model loss on training data
    val_loss : [int]
        Sequence of model loss on validation data
    val_metric : [float]
        Sequence of model metric on validation data
    """
    
    tr_loss = []
    val_loss = []
    val_metric = []
    
    # Main cycle over epochs
    for xepoch in range(num_epochs):
        if verbose:
            print('-'*32)
            print(f'Epoch {xepoch + 1}/{num_epochs}')
            print('-'*32)
        
        # Enable training mode, initialize loss accumulator
        model.train()
        loss_epoch = 0.0
        len_epoch = 0
        
        if show_progress:
            pbar = tqdm_notebook(total=len(train_loader), leave=False)
        
        # Cycle over mini-batches of data
        for (X, y) in train_loader:
            # Transport batch to device
            X_dev = X.to(device)
            y_dev = y.to(device)
            
            # Main part - cycle minibatch forward and backward and make
            # update step on parameters.
            # Just in case - zero gradient before
            opt.zero_grad()
            batch_pred = model(X_dev)
            batch_loss = loss_fct(batch_pred, y_dev)
            batch_loss.backward()
            opt.step()
            
            # Accumulate loss.
            # Loss comes out averaged over batch - restore original value
            loss_epoch += batch_loss.item()*len(X)
            len_epoch += len(X)
            
            if show_progress:
                pbar.update()
                
        # ================
        
        # Average loss over all training objects
        tr_loss.append(loss_epoch/len_epoch)
        
        if show_progress:
            pbar.close()
            
        if verbose:
            print(f'Training pass over. Loss: {tr_loss[-1]}')
        
        # Pass over validation data, get predictions and calcualate loss
        val_pred, val_gt, val_loss0 = predict_torch(model, val_loader, device)
        val_loss.append(val_loss0)
        
        # Get metric value
        if metric:
            val_metric.append(metric(
                val_gt.numpy(), val_pred.numpy().argmax(axis=1)
            ))
            
        if verbose:
            print(f'Validation loss: {val_loss[-1]}')
            
            if metric:
                print(f'Validation metric: {val_metric[-1]}')
                
            print('\n')
        
        # Adjust learning rate
        try:
            lr_sch.step(metrics=val_loss0)
        except TypeError:
            lr_sch.step()
        
        # Throttling if enabled
        if delay > 0:
            if verbose:
                print(f'Delay for {delay} seconds')
                
            time.sleep(delay)
        
    # ================================================
    
    return tr_loss, val_loss, val_metric


def visualize_training(loss_tr, loss_val, acc_val):
    """
    Helper function to build graphs of training history
    
    loss_tr, loss_val, acc_val - lists/arrays with training/validation loss
    and validation accuracy
    """
    fig0, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    axs[0].plot(np.arange(len(loss_tr)), loss_tr, color='red', label='Train')
    axs[0].plot(np.arange(len(loss_tr)), loss_val, color='blue', label='Validation')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Logloss')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(np.arange(len(loss_tr)), 100*np.array(acc_val), color='red')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Validation accuracy, %')
    axs[1].grid(True)
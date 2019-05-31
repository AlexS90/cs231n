#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
if not '..' in sys.path:
    sys.path = ['..'] + sys.path

import torch
from torch import nn

from copy import deepcopy

from tqdm import tqdm_notebook

import numpy as np
import matplotlib.pyplot as plt
from metrics import accuracy

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
                num_epochs=10, lr_plateau=False, 
                early_stopping=False, early_stopping_rounds=3, init_val_loss=np.inf, 
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
    best_state : 
        Model weights with best metric on validation set
    """
    
    tr_loss = []
    val_loss = []
    val_metric = []
    
    best_val_loss = init_val_loss
    best_state = None
    
    rounds_no_improv = 0
    keep_running = True
    
    # Main cycle over epochs
    xepoch = 0
    while keep_running:
        if verbose:
            print('\n' + '-'*32)
            print(f'Epoch {xepoch + 1}/{num_epochs}')
            print('-'*32)
        
#            try:
#                print(f'Learning rate: {lr_sch.get_lr()[0]}')
#            except:
#                pass
        
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
            print(f'Training pass over.\nTraining loss: {tr_loss[-1]}')
        
        # Pass over validation data, get predictions and calcualate loss
        val_pred, val_gt, val_loss0 = predict_torch(model, val_loader, device)
        val_loss.append(val_loss0)
        
        if verbose:
            print(f'Validation loss: {val_loss[-1]}')
        
        # Record best parameters
        if val_loss0 < best_val_loss:
            best_val_loss = val_loss0
            best_state = deepcopy(model.state_dict())
            rounds_no_improv = max(rounds_no_improv - 1, 0)
        else:
            rounds_no_improv += 1
            
            if verbose:
                print(f'No improvement on validation for {rounds_no_improv} epochs')
            
            if early_stopping and (rounds_no_improv >= early_stopping_rounds):
                keep_running = False
                model.load_state_dict(best_state)
                
                if verbose:
                    print('Invoking early stopping')
        
        # Get metric value
        if metric:
            val_metric.append(metric(
                val_gt.numpy(), val_pred.numpy().argmax(axis=1)
            ))
            
            if verbose:
                print(f'Validation metric: {val_metric[-1]}')
                
        # Adjust learning rate
        if lr_plateau:
            lr_sch.step(val_loss0)
        else:
            lr_sch.step()
        
        xepoch += 1
        keep_running &= (xepoch < num_epochs)
        
        # Throttling if enabled
        if (delay > 0) and keep_running:
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
    
    
def visualize_prediction_confidence(predX, gt):
    flag_correct = predX.argmax(axis=1) == gt
    prob_max = predX.max(axis=1)
        
    fig0, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
    
    plt.sca(axs[0])
    plt.hist([prob_max[flag_correct], prob_max[~flag_correct]], 
                label=['Correct prediction', 'Incorrect prediction'], 
                range=(0.0, 1.0), density=True, 
                bins=np.arange(0.0, 1.0 + 1e-3, 0.025), rwidth=0.9)
    plt.xlabel('Top class probability')
    plt.ylabel('Relative density')
    plt.title('Model confidence')
    plt.grid(True)
    plt.legend()
    
    # --------------------------------
    
    acc_thresh = []
    nonclf_thresh = []
    thresholds = np.linspace(0.01, 0.99, 98 + 1)
    
    for xthr in thresholds:
        flag_sure = predX.max(axis=1) >= xthr
        acc_thresh.append(accuracy(predX[flag_sure].argmax(axis=1), gt[flag_sure]))
        nonclf_thresh.append(sum(~flag_sure)/predX.shape[0])
    
    plt.sca(axs[1])
    plt.plot(thresholds, acc_thresh, color='red', label='Accuracy')
    plt.plot(thresholds, nonclf_thresh, color='blue', label='Not classified')
    plt.xlabel('Decision threshold')
    plt.title('Model performance with decision threshold')
    plt.grid()
    plt.legend()
    
    
def visualize_confusion_matrix(cm, cats, cmap='rainbow', text_col='white'):
    fig0, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
    
    plt.sca(axs[0])
    plt.imshow(cm, cmap=cmap)
    for ipred in range(cm.shape[0]):
        for igt in range(cm.shape[1]):
            plt.text(ipred, igt, cm[ipred, igt], 
                     ha='center', va='center', color=text_col, fontsize=14)
            
    plt.xlabel('Ground truth', fontsize=14)
    plt.ylabel('Model prediction', fontsize=14)
    plt.xticks(range(len(cats)))
    plt.yticks(range(len(cats)))
    axs[0].set_xticklabels(cats, rotation='vertical', fontsize=12)
    axs[0].set_yticklabels(cats, fontsize=12)
    plt.title('Confusion matrix', fontsize=16)
    
    # ================
    
    plt.sca(axs[1])
    plt.imshow(cm, cmap=cmap)
    for ipred in range(cm.shape[0]):
        for igt in range(cm.shape[1]):
            plt.text(ipred, igt, np.round(100*cm[ipred, igt]/cm[:, igt].sum(), 1), 
                     ha='center', va='center', color=text_col, fontsize=14)
            
    plt.xlabel('Ground truth', fontsize=14)
    plt.ylabel('Model prediction', fontsize=14)
    plt.xticks(range(len(cats)))
    plt.yticks(range(len(cats)))
    axs[1].set_xticklabels(cats, rotation='vertical', fontsize=12)
    axs[1].set_yticklabels(cats, fontsize=12)
    plt.title('Normalized confusion matrix', fontsize=16)
    
    
def visualize_errors(predX, gt, images, cats, n_per_class=5):
    fig0, axs = plt.subplots(nrows=len(cats), ncols=n_per_class, 
                             figsize=(16, 2*len(cats)))
    for (igt, xcat) in enumerate(cats):
        #Pick random images where model mistakes
        error_idxs = np.nonzero((gt == igt) & (predX.argmax(axis=1) != gt))[0]
        
        if len(error_idxs) < n_per_class:
            idxs = np.random.permutation(error_idxs)
        else:
            idxs = np.random.choice(error_idxs, n_per_class, replace=False)
        
        for ((ipred, idx), xax) in zip(enumerate(idxs), axs[igt]):
            xax.imshow(images[idx])
            xax.set_xlabel('\n'.join([
                f'Correct: {xcat}', 
                'Prob: {0:.3f}'.format(predX[idx, igt])]), fontsize=12)
            xax.set_title('\n'.join([
                f'Predicted: {cats[predX[idx].argmax()]}', 
                'Prob: {0:.3f}'.format(predX[idx].max())]), fontsize=12)
            #xax.axis('off')
            
            xax.spines['top'].set_visible(False)
            xax.spines['bottom'].set_visible(False)
            xax.spines['left'].set_visible(False)
            xax.spines['right'].set_visible(False)
            
            xax.set_xticks([])
            xax.set_yticks([])
            
    plt.tight_layout(pad=0.0, h_pad=1.0, w_pad=0.0)
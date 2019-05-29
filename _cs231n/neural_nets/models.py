#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from tqdm import tqdm_notebook
from collections import OrderedDict


def softmax_predict(X, y=None):
    """Performs softmax normalization on X.
    If y is not None, returns cross-entropy loss and gradient as well.
    
    Parameters:
    ----------------
    X : np.ndarray
        Feature matrix (index 0 is batch index, index 1 is feature index)
    y : np.ndarray
        Vector of labels
        
    Returns:
    ----------------
    hard_out : np.ndarray
        Hard softmax predictions (class with greatest probability value)
    probs : np.ndarray
        Soft predictions (normalized probabilities)
    loss : float
        Cross-entropy loss value (None if y is None)
    gradX : np.ndarray
        Gradient of cross-entropy loss with respect to X (None if y is None)
    """
    
    # Normalize X over feature dimension to avoid e^(BIG_NUMBER)
    scores = np.exp(X - X.max(axis=1)[:, None])
    
    # Get probabilities and hard predictions
    probs = scores/(scores.sum(axis=1)[:, None])
    hard_out = np.argmax(probs, axis=1)
    
    if y is None:
        # No y - no loss and grad
        return hard_out, probs, None, None
    else:
        # Calculate loss and gradient
        loss = -np.log(probs[np.arange(X.shape[0]), y]).sum()/X.shape[0]
        grad = probs.copy()
        grad[np.arange(X.shape[0]), y] -= 1.0
        
        # Gradient is averaged over number of samples
        return hard_out, probs, loss, grad/X.shape[0]
    
    
# ================================================================
# ================================================================
# ================================================================
    

class Model():
    """Basic class for complex neural network component.
    Subclasses should override forward and backward functions.
    
    Attributes
    ----------------
    layers : [layers.Layer or Model]
        List of module layers. Note: Terminology is flawed, these might
        be both Layer and Model instances.
    """
    
    def __init__(self, layers):
        """Initializes model. Layers are supplied as list, and their parameters names
        are altered: each has additional layer name added
        
        Parameters:
        ----------------
        layers : [layers.Layer]
            List of model layers. Order is important for sequential models.
        """
        
        # Layers are stored as ordered dict
        self.layers = OrderedDict()
        
        # Alter each parameter name by adding layer name to it.
        for xlayer in layers:
            if xlayer.name in self.layers.keys():
                raise ValueError('Duplicate layer names encountered')
            else:
                self.layers.update({xlayer.name: xlayer})
                
                for xparam in xlayer.params():
                    xparam.name = '.'.join([xlayer.name, xparam.name])


    def forward(self, X, train_pass=True):
        """Performs forward pass of a model.
        To be implemented by a subclass.
        
        Parameters:
        ----------------
        X : np.ndarray
            Feature matrix (index 0 is batch index)
        train_pass : bool
            True if forward pass results are needed or backprop. 
            If False - applicable layers won't store necessary info, and yet
            other applicable layers (e.g. Dropouts) will be excluded from computation.
        
        Returns:
        ----------------
        Y : np.ndarray
            Model output
        """
        
        pass


    def backward(self, grad_Y):
        """Performs backward pass of a model.
        To be implemented by a subclass.
        
        Parameters:
        ----------------
        grad_Y : np.ndarray
            Gradient of target function with respect to model output 
            (index 0 is batch index)
            
        Returns:
        ----------------
        grad_X : np.ndarray
            Gradint of target function with respect to model input.
            Also calculates gradients of all model parameters
        """
        
        pass
    
    
    def predict(self, X, y=None, out_fct=softmax_predict, 
                train_pass=False, batch_size=64):
        """Performs forward pass of a model coupled with classfier/loss function.
        Intended for inference, hence train_pass is False by default.
        Data is swooped in batches.
        
        Parameters:
        ----------------
        X : np.ndarray
            Feature matrix (index 0 is batch index)
        y : np.ndarray
            Vector of labels
        out_fct : function
            Function to be applied to model output (and optionally labels).
            Default is softmax with CE loss
        train_pass : bool
            Whether this is a train pass and some intermediate info required later
            for backprop
        batch_size : int > 0
            Size of data batch fed to model
        """
        
        # Split data into batches
        idxs = np.array_split(np.arange(X.shape[0]), int(np.ceil(X.shape[0]/batch_size)))
        
        # Feed data to model batch by batch, concatenate and feed to output function
        model_out = np.vstack([self.forward(X[xidx], train_pass) for xidx in idxs])
        return out_fct(model_out, y)
        

    def params(self):
        """Returns a list of model trainable parameters.
        List is recursive - every parameter of every submodel is returned.
        
        Returns:
        ----------------
        model_params : [Parameter]
            List of all trainable model parameters
        """
        
        model_params = []
        
        for xlayer in self.layers.values():
            model_params += xlayer.params()
        
        return model_params


    def reg_loss(self):
        """Returns regularization loss of model trainable parameters.
        Loss is recursive - reg. loss of every parameter of every submodel is aggregated.
        
        Returns:
        ----------------
        loss : float
            Aggregated loss value of model parameters regularization 
        """
        
        loss = 0.0
        
        for xlayer in self.layers.values():
            loss += xlayer.reg_loss()
            
        return loss


    def reset_grad(self):
        """Sets gradients of all model parameters to zero.
        """
        for xlayer in self.layers.values():
            xlayer.reset_grad()


# ================================================================
            
    
class SequentialModel(Model):
    """Subclass of Model, container for storing layers that are meant to
    be evaluated in sequence
    """
    def __init__(self, layers):
        super(SequentialModel, self).__init__(layers)


    def forward(self, X, train_pass=True):
        """Output of each layer is fed to input of next layer.
        """
        Y = X.copy()
        
        for xlayer in self.layers.values():
            if train_pass:
                Y = xlayer.forward(Y, True)
            else:
                if xlayer.required_on_inference:
                    Y = xlayer.forward(Y, False)
                
        return Y


    def backward(self, grad_Y):
        """Output of each layer is fed to input of next layer in reverse order
        """
        grad_X = grad_Y.copy()
        
        for xlayer in reversed(self.layers.values()):
            grad_X = xlayer.backward(grad_X)
            
        return grad_X


# ================================================================
# ================================================================
# ================================================================
        
    
def train_SGD(model, fct_loss, 
              X_train, y_train, X_val=None, y_val=None, shuffle=True, 
              epochs=5, batch_size=64, metric=None, parameters=None, 
              lr=1e-3, lr_decrease_step=5, lr_decrease_coef=0.5, momentum=0.9, 
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
    parameters : list of Parameter objects
        references to parameters that should be optimized
    lr : float > 0
        learning rate for gradient descend
    lr_decrease_step : int > 0
        After each lr_decrease_step'th step learning rate will be decreased
    lr_decrease_coef : float < 0
        Coeeficient of learning rate decreasing
    momentum : 0 <= float <= 1
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
            pbar = tqdm_notebook(total=len(batch_idxs), leave=False)
        
        # --------------------------------
        # Cycle over batches. Necessary to reset gradients beforehands
        batch_loss = []
        model.reset_grad()
            
        for xbatch_idxs in batch_idxs:
            # Forward and loss gradient
            model_out = model.forward(X_train[xbatch_idxs, :], train_pass=True)
            loss, loss_grad = fct_loss(model_out, y_train[xbatch_idxs])[2:]
            batch_loss.append(loss)
            
            # Backward pass
            model.backward(loss_grad)
            
            # Recalculate gradient and make SGD step
            for (xparam, xgrad) in zip(parameters, param_grad):
                xgrad = momentum*xgrad + xparam.grad
                xparam.value -= lr*xgrad
                
            # Zero gradients
            model.reset_grad()
                
            if show_progress:
                pbar.update()
                
        # --------------------------------
        # Passed over batches - now accumulate epoch loss as average over batches
        
        if show_progress:
            pbar.close()
            pbar.refresh()
        
        loss_history_train.append(np.mean(batch_loss))
        
        # If there is a validation set - calculate loss on it
        if not X_val is None:
            val_pred = model.predict(X_val, y_val, train_pass=False)
            loss_history_val.append(val_pred[2])
            
            # If there is a metric - calculate
            if metric:
                metric_history.append(metric(y_val, val_pred[0]))
                
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
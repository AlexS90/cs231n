#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.sparse import csr_matrix
import itertools as it


# ================================================================
# ================================================================
# ================================================================


def reg_L2(reg_coef, par_value):
    """Returns value and gradient of parameter L2 regularization.
    L2 = coef*W**2
    
    Parameters:
    ----------------
    reg_coef : float
        Coefficient of regularization
    par_value : np.ndarray
        Matrix of parameter values
        
    Returns: 
    ----------------
    reg_loss : float
        Regularization loss value
    reg_grad : np.ndarray
        Regularization loss gradient with respect to parameter value.
        Same shape as par_value
    """
    
    reg_loss = reg_coef*np.sum(par_value**2)
    reg_grad = 2*reg_coef*par_value
    return (reg_loss, reg_grad)


def reg_L1(reg_coef, par_value):
    """Returns value and gradient of parameter L1 regularization.
    L1 = coef*abs(W)**2
    
    Parameters:
    ----------------
    reg_coef : float
        Coefficient of regularization
    par_value : np.ndarray
        Matrix of parameter values
        
    Returns: 
    ----------------
    reg_loss : float
        Regularization loss value
    reg_grad : np.ndarray
        Regularization loss gradient with respect to parameter value.
        Same shape as par_value
    """
    
    reg_loss = reg_coef*np.sum(np.abs(par_value))
    
    neg_mask = par_value < 0
    reg_grad = np.full_like(par_value, reg_coef)
    reg_grad[neg_mask] = -reg_grad[neg_mask]
    return (reg_loss, reg_grad)


class Parameter:
    """Placeholder for optimizable parameter.
    Stores value, gradient and regularization loss.
    
    Attributes
    ----------------
    name : str
        Parameter name. Might be inside Model for accessing
    value : np.ndarray
        Numpy array holding parameter value
    grad : np.ndarray
        Numpy array holding parameter gradient of some function with respect
        to parameter
    reg_coef : float
        Coefficient of regularization
    reg_fct : function float, np.ndarray -> float, np.ndarray
        Takes regularization coefficient and parameter value, returns regularization
        loss and gradient with respect to parameter
    """
    
    def __init__(self, name, init_value=None, reg_coef=0.0, reg_type=None):
        """Constructor
        
        Parameters:
        ----------------
        name : str
            Parameter name
        init_value : np.ndarray
            Value to initialize parameter. Shape is inferred from this value, 
            so if specific shape is need without any data - pass np.zeros(shape).
            If None - will be initialized as [0]
        reg_coef : float
            Coefficient of regularization
        reg_type : str or None
            If 'L2' or 'L1' - corresponding regularization will be used.
            Otherwise no regularization.
        """
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
        """Returns parameter regularization loss and gradient, based 
        on current value
        """
        
        return self.reg_fct(self.reg_coef, self.value)


# ================================================================
# ================================================================
# ================================================================


class Layer:
    """Base class for neural networks layers, i.e. objects that accept
    data matrix, transform it and keep track of actions so that later
    gradient of specific function with respect to data matrix might be
    calculated.
    
    Attributes:
    ----------------
    name : str
        Layer name. Might be used inside Model for accessing
    required on inference : bool
        Controls whether this layers must be passed through when model
        is not training (only inferencing). Usually all layers are needed, 
        but, for example, Dropout is omitted on inference.
    """
    
    
    def __init__(self, name, required_on_inference=True):
        """Constructor.
        
        Parameters:
        ----------------
        name : str
            Layer name.
        required on inference : bool
            Controls whether this layers must be passed through when model
            is not training (only inferencing). Usually all layers are needed, 
            but, for example, Dropout is omitted on inference.
        """
        
        self.name = name
        self.required_on_inference=required_on_inference
    
    
    def forward(self, X, accum_grad=True):
        """Interface for layer forward pass.
        Each subclass has to implement this.
        
        Parameters:
        ----------------
        X : np.ndarray
            Data matrix. Index 0 is for batch indexing.
        accum_grad : bool
            If True (default) - layer will accumulate information required
            for backward pass. Otherwise backward pass is impossible.
            
        Returns:
        ----------------
        Y : np.ndarray
            Layer output.
        """
        
        return None
    
    
    def backward(self, grad_Y):
        """Interface for layer backward pass.
        Each subclass has to implement this.
        Returns gradient with respect to layer input and also calculates gradients
        with respect to all layer parameters.
        
        Parameters:
        ----------------
        grad_Y : np.ndarray
            Gradient with respect to layer output.
            
        Returns:
        ----------------
        grad_X : np.ndarray
            Gradient with respct to layer input.
        """
        
        return None
    
    
    def params(self):
        """Returns list of all trainable parameters.
        Each subclass has to implement this if it has any parameters to train.
        
        Returns:
        ----------------
        : [Parameter]
            List of trainable parameters
        """
        
        return []
    
    
    def reg_loss(self):
        """Returns regularization loss of all trainable parameters.
        Generally doesnt have to be reimplemented
        
        Returns:
        ----------------
        : float
            Aggregated regularization loss for all trainable parameters.
        """
        
        return(sum(xparam.reg_loss()[0] for xparam in self.params()))
    
    
    def reset_grad(self):
        """Sets gradients of all trainable parameters to zero.
        Generally doesnt have to be reimplemented
        """
        
        return None


# ================================================================
# ================================================================
# ================================================================


class DenseLayer(Layer):
    """Dense (fully-connected) layer.
    Forward pass is linear transformation with shift:
    Y = X*W + B,
    where W is weight matrix, B is shift column vector.
    X must be n by m matrix, where index 0 is for batch indexing.
    
    Attributes:
    ----------------
    features_in, features_out : int > 0
        Number of features (columns) of input/output data matrix
    include_bias : bool
        Whether to use shift or not
    reg_coef : float
        Regularization coefficient. Note: only weight matrix W is regularized.
        B is not
    init_std : float
        Standard deviation for weight matrix W initiaization. If None, 
        2/sqrt(features_in) is used
    W : Parameter
        Weight matrix of shape (features_in, features_out). Trainable.
    B : Parameter
        Shift vector of shape (features_out, ). Trainable. Absent if include_bias
        is False
    X_fw : np.ndarray
        Snapshot of data matrix of the last forward pass. Used for backprop.
        Not recorded if accum_grad is False on forward pass.
    """
    
    def __init__(self, name, features_in, features_out, include_bias=True, 
                 reg_coef=0.0, init_std=None, required_on_inference=True):        
        super().__init__(name, required_on_inference)
        self.features_in = features_in
        self.features_out = features_out
        
        # ================
        # Initialize weight matrix
        
        if init_std is None:
            init_std = np.sqrt(2.0/features_in)
        
        self.W = Parameter('W', init_std*np.random.normal(size=(features_in, features_out)), 
                           reg_coef, 'L2')
        
        if include_bias:
            self.B = Parameter('B', np.zeros((features_out, )))
        
        # Placeholder for forward pass data matrix
        self.X_fw = None
            
    
    def forward(self, X, accum_grad=True):
        # Save datamatrix it will be needed for backward pass.
        if accum_grad:
            self.X_fw = X.copy()
            
        Y = np.dot(X, self.W.value)
        
        # If layer has bias - shift by it
        if hasattr(self, 'B'):
            Y += self.B.value[None, :]
            
        return Y
    
    
    def backward(self, grad_Y):
        # Check if input data is stored from forward pass
        if self.X_fw is None:
            raise ValueError('No arguments recorded over the last forward pass or they have been erased')
        
        # Input gradient is Y*W^T
        grad_X = np.dot(grad_Y, self.W.value.T)
        # W gradient is X^T*Y. Also + reg loss
        self.W.grad = np.dot(self.X_fw.T, grad_Y) + self.W.reg_loss()[1]
        
        # B gradient is sum over all batch objects
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
    """Rectified linear unit activation layer.
    Performs max(0, x) on forward pass.
    Has no trainable parameters.
    
    Attributes:
    ----------------
    neg_mask : np.ndarray
        bool flags of elements of input matrix that are below zero.
        Used for backprop.
    """
    
    def __init__(self, name, required_on_inference=True):
        super().__init__(name, required_on_inference)
        
        # Placeholder for negative mask
        self.neg_mask = None
        
        
    def forward(self, X, accum_grad=True):
        # Flags for negative values
        neg_mask = X < 0.0
        
        # Negative mask is needed for backprop
        if accum_grad:
            self.neg_mask = neg_mask
            
        Y = X.copy()
        Y[neg_mask] = 0.0
        
        return Y
    
    
    def backward(self, grad_Y):
        # Check if negative mask is stored from forward pass
        if self.neg_mask is None:
            raise ValueError('No arguments recorded over the last forward pass or they have been erased')
        
        # Gradients of zeroed inputs are also zero
        grad_X = grad_Y.copy()
        grad_X[self.neg_mask] = 0.0
        
        return grad_X
    
    
    def reset_grad(self):
        self.neg_mask = None


# ================================================================
# ================================================================
# ================================================================


class Conv2D(Layer):
    """Two-dimensional convolutional layer.
    Receives 2D image in the form of np.ndarray of shape (B, H, W, C), where:
        B - batch size, 
        H, W - height/width, 
        C - channels,
    and performs convolution with trainable filter W and shift B.
    H and W might vary, as long as they're compatible with filter size,
    but C must be fixed.
    
    Attributes:
    ----------------
    channels_in, channels_out : int
        Number of channels of incoming/outcoming matrix
    filter_size : int
        Spatial size of filter. Total filter size is 
        (filter_size, filter_size, channels_in, channels_out)
    stride : int
        Numnber of rows/columns filter moves by when performing convolution.
    padding : int
        How many rows/columns of zeros to add to training data.
        Increases effective H and W, used for preserving spatial shape.
    include_bias : bool
        Whether to include shift component in convolution.
    reg_coef : float
        Regularization coefficient for filter matrix.
    W : Parameter
        Convolution filter. Trainable.
        Matrix of shape (filter_size, filter_size, channels_in, channels_out)
    B : Parameter
        Convolution constant shift. Trainable.
        Matrix of shape (channels_out, )
    xshape : (int, int)
        Internal parameter. Records spatial dimensions (H, W) of data used on
        latest forward pass
    Yh, Yw : np.ndarray
        Internal parameter. Spatial coordinates of layer output elements.
    Xh, Xw, Xc : np.ndarray
        Internal parameter. Spatial and depth coordinates of layer input elements.
    step_Xh, step_Xw : np.ndarray
        Spatial coordinates of input data when convolving with respect to
        LU-corner of filter
    X_fw : np.ndarray
        Snapshot of data matrix of the last forward pass. Used for backprop.
        Not recorded if accum_grad is False on forward pass.
    """
    
    def __init__(self, name, channels_in, channels_out=1, 
                 filter_size=3, stride=1, padding=0, 
                 include_bias=True, reg_coef=0.0, required_on_inference=True):
        super().__init__(name, required_on_inference)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        # Initialize convolution filter weight matrix
        self.W = Parameter('W', 1/np.sqrt(filter_size**2*channels_in*channels_out)*\
            np.random.normal(size=(filter_size, filter_size, channels_in, channels_out)), 
            reg_coef=reg_coef, reg_type='L2')
        
        # Initialize shift vector
        if include_bias:
            self.B = Parameter('B', np.zeros((channels_out, )))
        
        # --------------------------------
        
        # Placeholders for internal parameters
        self.xshape = None
        
        # Increments for H, W, C of X when applying convolution filter
        tmp = np.array(list(it.product(
            range(self.filter_size), 
            range(self.filter_size), 
            range(self.channels_in)
        )))
        
        self.step_Xh = tmp[:, 0]
        self.step_Xw = tmp[:, 1]
        
        # As of now only Xc can be calculated, all other depend on image size
        self.Xh = None
        self.Xw = None
        self.Xc = np.array(list(range(channels_in))*filter_size**2)
        self.Yh = None
        self.Yw = None
        
        
    def ravel_X(self, X):
        """Helper function.
        Ravels X from (B, H, W, C) into (B, L) shape for efficient convolution.
        Each row represents whole image (i.e. slice (b, :, :, :)) raveled into
        1D vector.
        
        Image is raveled in the following manner: Filter starts at (0, 0), which
        gives first filter_size**2*channels_in elements. It them moves first by
        width, them by height.
        
        Naturally many elements will be repeated, unless stride is big enough
        so that there is no overlap between filter applications.
        """
        
        return X[:, self.Xh.ravel(), self.Xw.ravel(), np.tile(self.Xc.ravel(), self.Xh.shape[0])].\
                   reshape(X.shape[0], -1, self.filter_size**2*self.channels_in)
        
        
    def forward(self, X, accum_grad=True):
        # This block is building so-called locality map - mapping from
        # spatial coordinates of Y to coordinates of elements of X 
        # are used in calculation of convolution of each element of Y.
        # This is constant for constant image size, so normally it is
        # done only once on first call.
        # But if X shape doesnt match the shape for which locality map is build
        # it has to be recalculated.
        
        if (self.xshape is None) or (self.xshape != X.shape[1:3]):
            # Check if X shape is compatible with filter size and stride
            # Naturally, filter must have no overlaps with borders
            assert (X.shape[1] + 2*self.padding - self.filter_size)%self.stride == 0
            assert (X.shape[2] + 2*self.padding - self.filter_size)%self.stride == 0
            assert X.shape[3] == self.channels_in
            self.xshape = X.shape[1:3]
            
            # Calculate output shape. Indices are raveled as (H, W)
            # Padding is taken into account - the coordinates are
            # for padded matrix
            tmp = np.array(list(it.product(
                range((X.shape[1] + 2*self.padding - self.filter_size)//self.stride + 1), 
                range((X.shape[2] + 2*self.padding - self.filter_size)//self.stride + 1)
            )))
            
            self.Yh = tmp[:, 0].astype(np.uint16)
            self.Yw = tmp[:, 1].astype(np.uint16)
            
            # For each (Hy, Hw) these are coordinates (Hx, Hw) that are used
            # to calculate convolution
            self.Xh = np.array([self.stride*Yh + self.step_Xh for Yh in self.Yh]).astype(np.uint16)
            self.Xw = np.array([self.stride*Yw + self.step_Xw for Yw in self.Yw]).astype(np.uint16)
            
        # --------------------------------
        
        if accum_grad:
            self.X_fw = X
        
        # Complicated part
        # So filter is raveled into (whatever, channels_out)
        # X is raveled into (batch_size, whatever, whatever) - a 3D tensor
        # We need matrix prod of last 2 indices of X_raveled and W_raveled
        # Also, here is where we add zero padding
        Y = np.einsum('bij, jk -> bik', 
                      self.ravel_X(np.pad(X, ((0, 0), 
                                              (self.padding, self.padding), 
                                              (self.padding, self.padding), 
                                              (0, 0)), mode='constant')), 
                      self.W.value.reshape(-1, self.channels_out))
        
        # Now just shift each out channel by B
        if hasattr(self, 'B'):
            Y += self.B.value[None, None, :]
        
        # Index 1 if raveled Y represents spatial stricture
        # Unravel it back into 3D shape (4D with batch)
        return Y.reshape(X.shape[0], self.Yh[-1] + 1, self.Yw[-1] + 1, self.channels_out)
    
    
    def backward(self, grad_Y):
        # Check if gradient shape is compatible
        assert grad_Y.shape[1:] == (self.Yh[-1] + 1, self.Yw[-1] + 1, self.channels_out)
        
        # Use same trick - ravel Y into 2D shape so that we can treat it as
        # linear layer and calculate grad_X in raveled form
        grad_Y_raveled = grad_Y.reshape(grad_Y.shape[0], -1, self.channels_out)
        grad_X_raveled = np.einsum('bik, jk -> bij', 
                                   grad_Y_raveled, 
                                   self.W.value.reshape(-1, self.channels_out))
        
        # Placeholder for unraveled form
        grad_X = np.zeros((grad_Y.shape[0], 
                           self.xshape[0] + 2*self.padding, 
                           self.xshape[1] + 2*self.padding, 
                           self.channels_in))
        
        # Even more complicated part
        # Ok, so for each element of grad_X_raveled we have info about which
        # element of grad_X it corresponds to. Unless filter applications don't
        # overlap - almost every element of grad_X has several correspondenses
        # in grad_X_raveled - we just have to sum them
        # Tricky part is unravel indices in correct order.
        # First is depth, them raveled height-width, then batch
        grids = np.mgrid[:grad_Y.shape[0], :np.prod(self.Xh.shape)]
        np.add.at(grad_X, 
            (
                grids[0].ravel(), 
                self.Xh.ravel()[grids[1].ravel()], 
                self.Xw.ravel()[grids[1].ravel()], 
                np.tile(np.arange(self.channels_in), grad_X_raveled.size//self.channels_in)
            ), 
            grad_X_raveled.ravel()
        )
        
        # Fortunately, for W we can just use linear layer machinery
        # Just need to unravel in back into original shape afterwards
        self.W.grad = np.einsum('bkj, bki -> ij', 
                                grad_Y_raveled, 
                                self.ravel_X(np.pad(self.X_fw, 
                                             ((0, 0), 
                                             (self.padding, self.padding), 
                                             (self.padding, self.padding), 
                                             (0, 0)), mode='constant'))).\
            reshape(self.filter_size, self.filter_size, self.channels_in, self.channels_out) + \
            self.W.reg_loss()[1]
        
        # Same for B
        if hasattr(self, 'B'):
            self.B.grad = np.einsum('bki -> i', grad_Y_raveled)
        
        # Before returning - we need to cut away paddings of zeros, as we don't
        # need those gradients
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
    """Base class for two-dimensional pooling layer.
    Receives 2D image in the form of np.ndarray of shape (B, H, W, C), where:
        B - batch size, 
        H, W - height/width, 
        C - channels,
    and performs pooling with set aggregation function (most commonly max or avg)
    with receptive field of given size and stride. Preserves depth structure.
    Subclasses implement their own forward and backward passes.
    
    Attributes:
    ----------------
    channels_in, channels_out : int
        Number of channels of incoming/outcoming matrix
    pool_size : int
        Spatial size of filter.
    stride : int
        Number of rows/columns filter moves by when performing convolution.
    xshape : (int, int)
        Internal parameter. Records spatial dimensions (H, W) of data used on
        latest forward pass
    Yh, Yw : np.ndarray
        Internal parameter. Spatial coordinates of layer output elements.
    Xh, Xw  : np.ndarray
        Internal parameter. Spatial and depth coordinates of layer input elements.
    step_Xh, step_Xw : np.ndarray
        Spatial coordinates of input data when pooling with respect to
        LU-corner of filter
    X_fw : np.ndarray
        Snapshot of data matrix of the last forward pass. Used for backprop.
        Not recorded if accum_grad is False on forward pass.
    """
    
    def __init__(self, name, pool_size=2, stride=2, required_on_inference=True):
        super().__init__(name, required_on_inference)
        self.pool_size = pool_size
        self.stride = stride
        
        # --------------------------------
        # Placeholders for internal parameters
        
        self.xshape = None
        
        # Increments for H, W of X when applying convolution filter
        tmp = np.array(list(it.product(range(self.pool_size), range(self.pool_size))))
        self.step_Xh = tmp[:, 0]
        self.step_Xw = tmp[:, 1]
        
        self.Yh = None
        self.Yw = None
        self.Xh = None
        self.Xw = None
    
    
    def forward(self, X, accum_grad=True):
        # This block is building so-called locality map - mapping from
        # spatial coordinates of Y to coordinates of elements of X 
        # are used in calculation of pooling of each element of Y.
        # This is constant for constant image size, so normally it is
        # done only once on first call.
        # But if X shape doesnt match the shape for which locality map is build
        # it has to be recalculated.
        # This feature is common for any pooling, so it is implemented here
        
        if (self.xshape is None) or (self.xshape != X.shape[1:3]):
            # Check if X shape is compatible with filter size and stride
            # Naturally, filter must have no overlaps with borders
            assert (X.shape[1] - self.pool_size)%self.stride == 0
            assert (X.shape[2] - self.pool_size)%self.stride == 0
            
            self.xshape = X.shape[1:3]
            
            # Calculate output shape. Indices are raveled as (H, W)
            # Padding is taken into account - the coordinates are
            # for padded matrix
            tmp = np.array(list(it.product(
                range((self.xshape[0] - self.pool_size)//self.stride + 1), 
                range((self.xshape[1] - self.pool_size)//self.stride + 1)
            )))
            
            self.Yh = tmp[:, 0].astype(np.uint16)
            self.Yw = tmp[:, 1].astype(np.uint16)
            
            # For each (Hy, Hw) these are coordinates (Hx, Hw) that are used
            # to calculate convolution
            self.Xh = np.array([self.stride*Yh + self.step_Xh for Yh in self.Yh]).astype(np.uint16)
            self.Xw = np.array([self.stride*Yw + self.step_Xw for Yw in self.Yw]).astype(np.uint16)
            
        return None
    
    
    def ravel_X(self, X):
        """Helper function.
        Ravels X from (B, H, W, C) into (B, I, J, C) shape for efficient pooling.
        Each matrix represents one receptive field (i.e. slice 
        (b, i0:i1, j0:j1, c)) raveled into 1D vector.
        
        Image is raveled in the following manner: Filter starts at (0, 0), then 
        moves first by depth, then by width, then by height.
        
        Sometimes some elements will be repeated, unless stride is big enough
        so that there is no overlap between filter applications.
        """
        
        return X[:, self.Xh.ravel(), self.Xw.ravel(), :].\
                   reshape((X.shape[0], self.Xh.shape[0], self.Xh.shape[1], X.shape[3]))


# ----------------------------------------------------------------


class AveragePooling2D(Pooling2D):
    """Performs average 2D pooling of the image.
    Each output pixel is spatial average of pixels with receptive field
    """
    
    def __init__(self, name, pool_size=2, stride=2, required_on_inference=True):
        super().__init__(name, pool_size, stride, required_on_inference)
    
    
    def forward(self, X, accum_grad=True):
        super().forward(X, accum_grad)
        
        # Forward pass is simple here - just take average over each receptive field
        # It raveled into axis 2
        Y = self.ravel_X(X).mean(axis=2).\
            reshape((X.shape[0], self.Yh[-1] + 1, self.Yw[-1] + 1, X.shape[3]))
        
        return Y
    
    
    def backward(self, grad_Y):
        # Check if gradient shape is compatible
        assert grad_Y.shape[1:3] == (self.Yh[-1] + 1, self.Yw[-1] + 1)
        
        # Placeholder for input gradient, partially raveled
        grad_X = np.zeros((grad_Y.shape[0], np.prod(self.xshape), grad_Y.shape[3]))
        
        # Tricky part
        # Each element is raveled form corresponds to one element in unraveled form
        # Naturally some might overlap. Info about which goes to which is stored
        # in Xh, Xw - we just have to unravel indices correctly
        # First - depth
        # Second - H/W, raveled in one
        # Last - batch
        np.add.at(grad_X, 
            np.ix_(
                np.arange(grad_Y.shape[0]), 
                np.ravel_multi_index([self.Xh.ravel(), self.Xw.ravel()], self.xshape), 
                np.arange(grad_Y.shape[3])
            ), 
            np.repeat(grad_Y.reshape(grad_Y.shape[0], -1, grad_Y.shape[3]), 
                      self.Xh.shape[1], axis=1)
        )
        
        # Now unravel H/W into separate axes, and also divide by number of elements
        # in receptive field (i.e. take average)
        return grad_X.reshape(grad_Y.shape[0], *self.xshape, grad_Y.shape[3])/self.pool_size**2


# ----------------------------------------------------------------


class MaxPooling2D(Pooling2D):
    """Performs maximum 2D pooling of the image.
    Each output pixel is spatial maximum of pixels with receptive field
    
    Parameters:
    ----------------
    max_idx : np.ndarray
        Stores info about which element is selected for each receptive field.
        Needed for backprop
    """
    
    def __init__(self, name, pool_size=2, stride=2, required_on_inference=True):
        super().__init__(name, pool_size, stride, required_on_inference)
        
        # Placeholder for maximum mask
        self.max_idx = None
        
    
    def forward(self, X, accum_grad=True):
        super().forward(X, accum_grad)
        
        # Receptive field are raveled into axis 2
        # It is very easy to get indices
        # Unfortunately, numpy can't give us indices and maximum values in one
        # So we have to unravel index back to get the values
        X_raveled = self.ravel_X(X)
        self.max_idx = X_raveled.argmax(axis=2)
        
        # Indices of maximum elements
        grids = np.mgrid[:X.shape[0], :X_raveled.shape[1], :X.shape[3]]
        Y = X_raveled[grids[0].ravel(), 
                      grids[1].ravel(), 
                      self.max_idx.ravel(), 
                      grids[2].ravel()].\
            reshape(X.shape[0], self.Yh[-1] + 1, self.Yw[-1] + 1, X.shape[3])
        
        return Y
    
    
    def backward(self, grad_Y):
        # Check if gradient shape is compatible
        assert grad_Y.shape[1:3] == (self.Yh[-1] + 1, self.Yw[-1] + 1)
        
        # Placeholder for input gradient, partially raveled
        grad_X = np.zeros((grad_Y.shape[0], np.prod(self.xshape), grad_Y.shape[3]))
        
        # Tricky part
        # Each element is raveled form corresponds to one element in unraveled form
        # Naturally some might overlap. Info about which goes to which is stored
        # in Xh, Xw - we just have to unravel indices correctly
        # However, since not every element from Xh, Xw is needed - we filter
        # them in Xhmax, Xwmax
        Xhw_idx = np.tile(np.repeat(np.arange(self.Xh.shape[0]), grad_Y.shape[3]), grad_Y.shape[0])
        Xhmax = self.Xh[Xhw_idx, self.max_idx.ravel()]
        Xwmax = self.Xw[Xhw_idx, self.max_idx.ravel()]
        grids = np.mgrid[:grad_Y.shape[0], :self.Xh.shape[0], : grad_Y.shape[3]]
        
        np.add.at(grad_X, 
            (
                grids[0].ravel(), 
                np.ravel_multi_index([Xhmax, Xwmax], self.xshape), 
                grids[2].ravel()
            ), 
            grad_Y.ravel()
        )
        
        # Reshape back
        return grad_X.reshape(grad_Y.shape[0], *self.xshape, grad_Y.shape[3])
    
    
    def reset_grad(self):
        self.X_idxmax = None

# ----------------------------------------------------------------

# ================================================================
# ================================================================
# ================================================================


class Resizer(Layer):
    """Layer for rearranging spatial structure of data.
    No transformation is applied - only reshaping. 
    
    Attributes:
    ----------------
    shape_in, shape_out : (int, )
        Shapes of data on input/output. Must match on the number of elements.
        Batch index is not included.
    """
    
    def __init__(self, name, shape_in, shape_out=None, required_on_inference=True):
        super().__init__(name, required_on_inference)
        
        # Check that both shapes have same number of elements
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
    """Dropout layer.
    Randomly zeroes output of each neuron with given probability. 
    This brings stochasticity and reduces overfitting, as on a given pass,
    some neurons won't receive a gradient.
    This layer MUST be bypasses during inference, hence required_on_inference
    is False by default.
    
    Attributes:
    ----------------
    p_drop : 0 <= float <= 1.0
        Probability of dropping a neuron output
    dropped_mask : np.ndarray
        Internal parameter. Mask of which neurons are dropped. Used for backprop.
    """
    
    def __init__(self, name, p_drop=0.5, required_on_inference=False):
        super().__init__(name, required_on_inference)
        self.p_drop = p_drop
        
        # Placeholder for internal parameters
        self.dropped_mask = None
        
        
    def forward(self, X, accum_grad=True):
        # Select neurons to drop
        if accum_grad:
            self.dropped_mask = np.random.choice([False, True], X.shape, 
                                                 p=(1 - self.p_drop, self.p_drop))
        
        Y = X.copy()
        Y[self.dropped_mask] = 0.0
        
        # Normalizing output to ensure same average activation with and
        # without dropping
        return Y/(1 - self.p_drop)
    
    
    def backward(self, grad_Y):
        grad_X = grad_Y.copy()
        grad_X[self.dropped_mask] = 0.0
        
        # Normalizing output to ensure same average activation with and
        # without dropping
        return grad_X/(1 - self.p_drop)
    
    
    def reset_grad(self):
        self.dropped_mask = None


# ----------------------------------------------------------------


class BatchNorm(Layer):
    """Batch normalization layer.
    Scales input to 0 mean and 1 std and performs linear transformation
    of normalized input.
    This is simple BatchNorm, meaning that input is scaled by global mean/std
    (not per depth/dimension - two single numbers).
    
    Attributes:
    ----------------
    eps : float << 1.0
        Variance tweaker, added to batch variance to ensure no zero division
    scale, offset : Parameters
        Scale and shift of linear transformation on normalized data.
        Trainable
    X_cent_fw : np.ndarray
        Internal parameter. Normalized input snapshot. Used for backprop.
    X_batch_fw : np.ndarray
        Internal parameter. Batch variance. Used for backprop.
    """
    
    def __init__(self, name, required_on_inference=True, eps=1e-6):        
        super().__init__(name, required_on_inference)
        self.eps = eps
        
        # Initialize scale and offset
        self.scale = Parameter('scale', np.array([1.0]))
        self.offset = Parameter('offset', np.array([0.0]))
        
        # Placeholders for internal parameters
        self.X_cent_fw = None
        self.batch_var_fw = None
        
        
    def forward(self, X, accum_grad=True):
        # Calculate batch statistics and normalize batch
        batch_mean = X.mean()
        batch_var = X.var()
        X_cent = (X - batch_mean)/np.sqrt(batch_var + self.eps)
        
        # If training pass - remember forward parameters
        if accum_grad:
            self.X_cent_fw = X_cent
            self.batch_var_fw = batch_var
        
        # Linear transformation of normalized data
        return self.scale.value*X_cent + self.offset.value
    
    
    def backward(self, grad_Y):
        # Describing how gradients are calculated is tedious here
        # This is not convolution - nothing too special, easily googled
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from tqdm import tqdm

# ================================
# ================================
# ================================

def gradient_numerical(fct, X0, rel_delta_X=1e-6) -> np.ndarray:
    """
    Performs numerical computation of function gradient at given point
    
    Parameters:
    ----------------
    fct : function(np.ndarray) -> float
        Function mapping NumPy n-dimensional array to float.
        Gradient of this function with respect to input is calculated and returned.
    X0 : np.ndarray
        Point to compute gradient at, shape compatible with fct.
    delta_X : float > 0
        Step size to be used to compute gradient components.
        
    Returns:
    ----------------
    grad_numerical : np.ndarray
        Numerical estimation of gradient of fct with respect to input at X0
    """
    
    # Initialize gradient with zeros
    grad_numerical = np.zeros_like(X0)
    
    # Iterating through all elements of X0 one-by-one, computing gradient components    
    idx_iterator = np.nditer(X0, flags=['multi_index'])
    
    while not idx_iterator.finished:
        idx = idx_iterator.multi_index
        delta_X = rel_delta_X*abs(X0[idx])
        # Step forward
        X_plus = X0.copy()
        X_plus[idx] += delta_X
        
        # Step backward
        X_minus = X0.copy()
        X_minus[idx] -= delta_X
        
        # Computing gradient component
        grad_numerical[idx] = (fct(X_plus) - fct(X_minus))/(2*delta_X)
        idx_iterator.iternext()
        
    return grad_numerical


def compare_matrices(A, B, rel_tol=1e-6, verbose=True) -> bool:
    """
    Performs element-wise comparison of two arrays and returns True if
    all elements are within relative tolerance
    
    Parameters:
    ----------------
    A, B : np.array
        Matrices to be compared
    rel_tol : float > 0
        Required relative tolerance to deem matrices equal
    verbose : bool
        If True, a comprehensive report on every mismatching element will
        be printed
        
    Returns:
    ----------------
    res : bool
        True if all elements of both matrices are within required tolerance
    """
    
    assert A.shape == B.shape
    
    res = np.allclose(A, B, rtol=rel_tol, atol=0.0)
    
    if not res:
        print(f'Matrices are inconsistent within relative tolerance of {rel_tol}')
        if verbose:
            idx_iterator = np.nditer(A, flags=['multi_index'])
            
            while not idx_iterator.finished:
                idx = idx_iterator.multi_index
                
                if not np.isclose(A[idx], B[idx], rtol=rel_tol, atol=0.0):
                    rel_err = abs(A[idx] - B[idx])/abs(B[idx])
                        
                    print('Elements at {0} differ by abs. / rel. {1} / {2}'.format(
                        idx, abs(A[idx] - B[idx]), rel_err
                    ))
                    
                idx_iterator.iternext()
    else:
        print(f'Matrices are consistent within relative tolerance of {rel_tol}')
        
    print('\n')
        
    return res


def check_model_gradient(model, X0, out_fct=None, 
                         rel_delta_X=1e-6, reltol=1-6) -> bool:
    """
    Performs comparison of model-defined analytical gradient calculation and
    numerical computation within given tolerance. Used to check gradient calc
    implementation.
    
    Gradients are computed at point X0 with respect to model inputs and every
    parameter as well.
    
    Parameters:
    ----------------
    model : Layer or Model subclass instance
        Model to be checked. Must be an instance of Model or Layer subclass, i.e.
        implement functions forward, backward, params.
    X0 : np.ndarray
        Model input to compute gradients at.
    out_fct : function np.ndarray -> (float, np.ndarray)
        Since model output is NumPy array, potentiall multidimensional, to ease up
        computations an aggregation function is needed. out_fct takes model output
        and returns a single value and it's gradient with respect to model output 
        as a 2-element tuple.
        If not provided, summation with random coefficients will be used.
    delta_X : float > 0
        Step size to be used to compute gradient components.
    reltol : float > 0
        Relative error to check proximity of numerical and analytical gradients
        
    Returns:
    ----------------
    res : bool
        True if model input and all parameters have analytical and numerical gradients
        close within 1e-6 tolerance, False otherwise
    """
    
    # If aggregation function not provided - summation wit random coefficients will be used
    # It's gradient is simply a matrix of weights
    if out_fct is None:
        out_weights = np.random.normal(0, 1, size=model.forward(X0, train_pass=False).shape)
        out_fct = lambda Z: (np.sum(Z*out_weights), out_weights)

    # ================
    # First - check model input gradient
    
    # Perform forward and backward passes to calculate parameters gradients
    _, grad_y0 = out_fct(model.forward(X0))
    grad_X0_model = model.backward(grad_y0)
    
    # Compute gradients numerically 
    # Necessary to redefine out_fct to accept model input and return a single value
    grad_X0_num = gradient_numerical(lambda X: out_fct(model.forward(X))[0] + model.reg_loss(), 
                                     X0, rel_delta_X)
    
    # Check if gradients are within tolerance
    print('Checking model input...')
    res = compare_matrices(grad_X0_model, grad_X0_num, reltol, True)
    
    # ================
    # Now check every parameter
    # Use model interface to generate list of parameters
    
    for xparam in model.params():
        print(f'Checking parameter {xparam.name}')
        
        # Memorize parameter value prior to messing with it
        param_value0 = xparam.value
        
        # Now the function is dependent of parameter value, while model input is fixed
        # Set new parameter value explicitly and make forward pass
        def fct(xpar):
            xparam.value = xpar
            return out_fct(model.forward(X0, train_pass=False))[0] + model.reg_loss()
        
        # Compute gradient numerically and compare with analytical result
        # NOTE: Analytical result is calculated above within backeard pass
        # Despite all the consequnt forward passes, no new backward passes
        # performed -> gradients remain unchanged
        grad_param_num = gradient_numerical(fct, param_value0, rel_delta_X)
        res &= compare_matrices(xparam.grad, grad_param_num, reltol, True)
        xparam.value = param_value0
    
    # ================
    
    return res
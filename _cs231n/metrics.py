#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submodule for cs231n course, containing metrics implementation
"""

import numpy as np

# ================================
# ================================
# ================================

def clf_score(gt, pred) -> np.ndarray:
    """
    Returns confusion matrix of classification on a set of objects, specifically
    a matrix where element [i, j] is how many objects of class j got label i.
    NOTE: number of classes is inferred from ground truth labels
    
    Parameters:
    ----------------
    gt : 1D np.ndarray[int]
        Ground truth, array containing true labels of objects being classified
    pred : 1D np. ndarray of int
        Model predictions of objects being classified
        
    Returns:
    ----------------
    cm : 2D np.ndarray[int]
        Confusion matrix, element [i, j] equals to number of objects of true class j 
        and predicted class i.
    """
    
    # Number of classes inferred from gt. Assuming classes are enumerated 0 ..
    n_classes = gt.max()
    cm = np.zeros((n_classes, n_classes), dtype=np.uint32)
    
    # Fill matrix
    for gt_class in range(n_classes):
        for pred_class in range(n_classes):
            cm[pred_class, gt_class] = ((pred == pred_class) & (gt == gt_class)).sum()
            
    return cm


def clf_scores(cm) -> float:
    """
    Returns overall and per-class classification scores based on confusion matrix
    Scores returned are:
        - accuracy
        - precision
        - recall
        - F1
        
    Accuracy is calculated overall, other metrics - per each class
    
    Parameters:
    ----------------
    cm : square np.ndarray[int]
        Confusion matrix, element [i, j] equals to number of objects of true class j 
        and predicted class i.
    
    Returns:
    ----------------
    scores : dict of {str: (float, np.array[float])}
        Dict with overall and per-class scores.
    """
    
    # Number of correct elements of given class - diagonal
    acc = cm.trace()/np.sum(cm)
    
    # Total number of elements with true class j - sum of column j
    # Total number of elements with predicted class i - sum of row i
    precision_class = cm.diagonal()/cm.sum(axis=1)
    recall_class = cm.diagonal()/cm.sum(axis=0)
    f1_class = 2*precision_class*recall_class/(precision_class + recall_class)
    
    return {
        'accuracy': acc, 
        'precision': precision_class, 
        'recall': recall_class, 
        'F1': f1_class
    }
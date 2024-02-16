"""
Utility functions for the AutoCV.
"""

from sklearn.base import BaseEstimator

def is_sklearn_model(model):
    """
    Check if the model is a scikit-learn model.
    """
    try:
        from sklearn.base import is_classifier, is_regressor
        return isinstance(model, BaseEstimator) or is_classifier(model) or is_regressor(model)
    except ImportError:
        return False

def is_torch_model(model):
    """
    Check if the model is a PyTorch model.
    """
    try:
        import torch.nn as nn
        return isinstance(model, nn.Module) or (hasattr(model, 'parameters') and hasattr(model, 'forward'))
    except ImportError:
        return False

def is_tf_model(model):
    """
    Check if the model is a TensorFlow model.
    """
    try:
        import tensorflow as tf
        return isinstance(model, tf.Module) or isinstance(model, tf.keras.Model) or hasattr(model, 'fit')
    except ImportError:
        return False

def check_estimator_type(estimator):
    """
    Determine the type of the estimator.
    Returns 'nn' for neural network models, 'ml' for machine learning models.
    """
    if is_torch_model(estimator) or is_tf_model(estimator):
        return 'nn'
    elif is_sklearn_model(estimator):
        return 'ml'
    else:
        raise TypeError("The type of the estimator cannot be determined!")

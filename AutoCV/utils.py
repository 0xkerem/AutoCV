"""
Utility functions for the AutoCV.
"""

from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target

def determine_problem_type(target, detailed=False):
    """
    Determine if the problem is a classification or regression problem.

    Parameters:
    target: array-like, shape (n_samples,), the target variable.
    detailed: bool, optional (default=False). If True, return specific classification type.

    Returns:
    str: One of 'classification', 'binary_classification', 'multiclass_classification', or 'regression'.
    """
    target_type = type_of_target(target)
    
    if target_type == 'binary':
        return 'binary_classification' if detailed else 'classification'
    elif target_type == 'multiclass':
        return 'multiclass_classification' if detailed else 'classification'
    elif target_type in ['continuous', 'continuous-multioutput']:
        return 'regression'
    elif target_type in ['multilabel-indicator', 'multiclass-multioutput']: # TODO: Handle this types
        raise ValueError("Unsupported target type: {}".format(target_type))
    else:
        raise ValueError("Unsupported or unrecognized target type: {}".format(target_type))


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

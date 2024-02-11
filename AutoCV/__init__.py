"""
AutoCV: An automated cross-validation framework for machine learning models.
"""

__version__ = "0.1.0"
__author__ = "Kerem"
__email__ = "keremozrtk@gmail.com"

from .core import AutoCV
from .core import cross_val_score

__all__ = [
    "AutoCV",
    "cross_validate",
    "evaluate_performance",
    "BaseModel",
    "LinearModel",
    "TreeModel",
    "NeuralNetwork",
    "load_data",
    "preprocess_data",
]

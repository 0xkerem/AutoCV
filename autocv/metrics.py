"""
Default metrics for AutoCV.
"""

from .utils import determine_problem_type
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Define default scorers for binary classification
binary_classification_scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='binary', zero_division=1),
    'recall': make_scorer(recall_score, average='binary', zero_division=1),
    'f1_score': make_scorer(f1_score, average='binary', zero_division=1)
}

# Define default scorers for multi-class classification
multi_class_classification_scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro', zero_division=1),
    'recall': make_scorer(recall_score, average='macro', zero_division=1),
    'f1_score': make_scorer(f1_score, average='macro', zero_division=1)
}

# Define default scorers for regression
regression_scorers = {
    'mse': make_scorer(mean_squared_error),
    'mae': make_scorer(mean_absolute_error),
    'r2': make_scorer(r2_score)
}


def set_default_scoring(target):
    """
    Set default scoring metrics based on the type of problem.

    Parameters:
    target: array-like, shape (n_samples,), the target variable.

    Returns:
    dict: A dictionary of scoring functions appropriate for the problem type.
    """
    problem_type = determine_problem_type(target, detailed=True)

    if problem_type == 'binary_classification':
        return binary_classification_scorers
    elif problem_type == 'multiclass_classification':
        return multi_class_classification_scorers
    elif problem_type == 'regression':
        return regression_scorers
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

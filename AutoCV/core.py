"""
Core module for the AutoCV framework.
"""

import numpy as np
from sklearn.model_selection import (
    cross_validate as sklearn_cv,
    StratifiedKFold,
    KFold,
    GroupKFold
)
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder

class AutoCV:
    def __init__(self, model, cv=None, scoring=None, group_column=None):
        """
        Initializes the AutoCV object.

        Parameters:
        model: A machine learning model instance (e.g., from scikit-learn)
        cv: int, cross-validation splitting strategy
        scoring: str or callable, a scoring method to evaluate the performance (default is None)
        use_group_kfold: bool, whether to use Group K-Fold cross-validation (default is False)
        group_column: array-like, group labels for Group K-Fold cross-validation (default is None)
        """
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.group_column = group_column

    def _determine_problem_type(self, y):
        """
        Determine if the problem is a classification or regression problem.

        Parameters:
        y: array-like, shape (n_samples,), the target variable

        Returns:
        str: 'classification' or 'regression'
        """
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass']:
            return 'classification'
        elif target_type in ['continuous', 'continuous-multioutput']:
            return 'regression'
        else:
            raise ValueError("Unsupported target type: {}".format(target_type))

    def _is_imbalanced(self, y):
        """
        Check if the classification data is imbalanced.

        Parameters:
        y: array-like, shape (n_samples,), the target variable

        Returns:
        bool: True if the data is imbalanced, False otherwise
        """
        class_counts = np.bincount(y)
        minority_class_ratio = np.min(class_counts) / np.sum(class_counts)
        return minority_class_ratio < 0.3

    def cross_validate(self, X, y):
        """
        Perform automatic cross-validation.

        Parameters:
        X: array-like, shape (n_samples, n_features), the input data
        y: array-like, shape (n_samples,), the target variable

        Returns:
        dict: A dictionary containing the mean and standard deviation of the cross-validation scores.
        """
        problem_type = self._determine_problem_type(y)

        if problem_type == 'classification':
            # Encode labels if they are not numeric
            if y.dtype == 'O' or isinstance(y[0], str):
                y = LabelEncoder().fit_transform(y)
            
            # Check if data is imbalanced
            if self._is_imbalanced(y):
                print("Data is imbalanced. Using Stratified K-Fold.")
                cv_strategy = StratifiedKFold(n_splits=self.cv)
            else:
                cv_strategy = KFold(n_splits=self.cv)
        else:
            cv_strategy = KFold(n_splits=self.cv)

        # Use Group K-Fold if specified
        if self.use_group_kfold:
            if self.group_column is not None:
                cv_strategy = GroupKFold(n_splits=self.cv)
                groups = self.group_column
            else:
                raise ValueError("Group column must be provided for Group K-Fold cross-validation.")
        else:
            groups = None

        print("Starting cross-validation...")  # Debug statement
        scores = sklearn_cv(self.model, X, y, cv=cv_strategy, scoring=self.scoring, groups=groups)
        print(f"Scores: {scores}")  # Debug statement
        results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
        print(f"Results: {results}")  # Debug statement
        return results
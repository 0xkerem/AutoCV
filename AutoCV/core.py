"""
Core module for the AutoCV framework.
"""

import numpy as np
from sklearn.model_selection import (
    cross_validate as sklearn_cv,
    StratifiedKFold,
    KFold,
    GroupKFold,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    StratifiedShuffleSplit
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
    
    def _determine_n_splits(self):
        pass

    def cross_validate(self, X, y, force=False):
        """
        Perform automatic cross-validation.

        Parameters:
        X: array-like, shape (n_samples, n_features), the input data
        y: array-like, shape (n_samples,), the target variable
        force: bool, optional (default=False), If True, forces cross-validation on large datasets.

        Returns:
        dict: A dictionary containing the mean and standard deviation of the cross-validation scores.
        """
        size = X.shape[0]

        if size > 100000 and not force:
            print("Dataset is large. Skipping cross-validation. Set `force=True` to override.")
            return None
        
        problem_type = self._determine_problem_type(y)

        if size <= 200:
            cv_strategy = LeaveOneOut()
        elif size <= 1000:
            cv_strategy = LeavePOut(p=np.round(size/100))
        elif problem_type == 'classification':
            # Encode labels if they are not numeric
            if y.dtype == 'O' or isinstance(y[0], str):
                y = LabelEncoder().fit_transform(y)
            
            # Check if data is imbalanced
            if self._is_imbalanced(y):
                print("Data is imbalanced. Using Stratified K-Fold.")
                if size < 20000:
                    cv_strategy = StratifiedKFold(n_splits=self.cv)
                else:
                    cv_strategy = StratifiedShuffleSplit(n_splits=self.cv)
        else:
            if size < 20000:
                cv_strategy = KFold(n_splits=self.cv)
            else:
                cv_strategy = ShuffleSplit(n_splits=self.cv)

        # Use Group K-Fold if specified
        if self.group_column is not None:
            cv_strategy = GroupKFold(n_splits=self.cv)
            groups = self.group_column

        scores = sklearn_cv(self.model, X, y, cv=cv_strategy, scoring=self.scoring, groups=groups)
        results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
        return results
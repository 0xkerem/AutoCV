"""
Core module for the AutoCV.
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
from sklearn.preprocessing import LabelEncoder
from utils import (
    check_estimator_type,
    determine_problem_type
)
from metrics import set_default_scoring


LARGE_LIMIT = 20000


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
        self.estimator_type = None


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
    

    def _determine_n_splits(self, size: int):
        """
        Determine the number of cross-validation splits based on the dataset size.

        Parameters:
        size: int, the number of samples in the dataset

        Raises:
        ValueError: If the number of splits is greater than the number of data points.
        """
        if self.cv is None:
            # Set default number of splits based on dataset size and estimator type
            if size <= 2500 or size > LARGE_LIMIT:
                self.cv = 10
            elif size > 10000 and self.estimator_type == 'nn':
                self.cv = 3
            else:
                self.cv = 5
        elif self.cv > size:
            raise ValueError(f"Number of splits (cv={self.cv}) cannot be more than the number of data points (n={size}).")


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

        # Check if the dataset is too large
        if size > 100000 and not force:
            print("Dataset is large. Skipping cross-validation. Set `force=True` to override.")
            return None

        self.estimator_type = check_estimator_type(self.model)

        self._determine_n_splits(size)

        # Determine the type of the estimator
        problem_type = determine_problem_type(y)

        # Determine the cross-validation strategy
        cv_strategy = self._select_cv_strategy(size, problem_type, y)

        # Use Group K-Fold if specified
        if self.group_column is not None:
            cv_strategy = GroupKFold(n_splits=self.cv)
            groups = self.group_column
        else:
            groups = None

        # Use default scoring if is is not set manually
        if self.scoring is None:
            self.scoring = set_default_scoring(y)

        # Perform cross-validation
        scores = sklearn_cv(self.model, X, y, cv=cv_strategy, scoring=self.scoring, groups=groups)
        results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
        # TODO: Return direct to average score but create new datatype to save other informations
        return results


    def _select_cv_strategy(self, size, problem_type, y):
        """
        Select the appropriate cross-validation strategy based on the dataset size and problem type.

        Parameters:
        size: int, the number of samples in the dataset
        problem_type: str, the type of problem (e.g., 'classification' or 'regression')
        y: array-like, shape (n_samples,), the target variable

        Returns:
        cv_strategy: cross-validation strategy object
        """
        if size <= 200:
            return LeaveOneOut()
        elif size <= 1000:
            return LeavePOut(p=np.round(size / 100))
        elif problem_type == 'classification':
            y = self._encode_labels_if_needed(y)
            if self._is_imbalanced(y):
                if size < LARGE_LIMIT:
                    return StratifiedKFold(n_splits=self.cv)
                else:
                    return StratifiedShuffleSplit(n_splits=self.cv)
            else:
                if size < LARGE_LIMIT:
                    return KFold(n_splits=self.cv)
                else:
                    return ShuffleSplit(n_splits=self.cv)
        else:
            if size < LARGE_LIMIT:
                return KFold(n_splits=self.cv)
            else:
                return ShuffleSplit(n_splits=self.cv)


    def _encode_labels_if_needed(self, y):
        """
        Encode labels if they are not numeric.

        Parameters:
        y: array-like, shape (n_samples,), the target variable

        Returns:
        y: array-like, shape (n_samples,), the encoded target variable
        """
        if y.dtype == 'O' or isinstance(y[0], str):
            return LabelEncoder().fit_transform(y)
        return y

    # TODO: Create summary method
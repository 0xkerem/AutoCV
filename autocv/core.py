"""
Core module for the AutoCV.
"""

import numpy as np
from .metrics import set_default_scoring
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import (
    cross_validate as sklearn_cv,
    StratifiedKFold,
    KFold,
    GroupKFold,
    StratifiedGroupKFold,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    StratifiedShuffleSplit
)

from .utils import (
    check_estimator_type,
    determine_problem_type,
    Result
)


class AutoCV:
    def __init__(self, model, n_splits=None, scoring=None, group_column=None, random_state=None):
        """
        Initializes the AutoCV object.

        Parameters:
        model: A machine learning model instance (e.g., from scikit-learn)
        n_splits: int, cross-validation splitting strategy (default is None)
        scoring: str or callable, a scoring method to evaluate the performance (default is None)
        group_column: array-like, group labels for Group K-Fold cross-validation (default is None)
        random_state: int or RandomState instance, controls the randomness of the estimator (default is None)

        Attributes:
        estimator_type: Type of the estimator, determined after fitting the model
        result: Results of the cross-validation
        cv_strategy: Strategy used for cross-validation, determined based on input parameters
        __large_limit: Internal parameter to manage computational complexity thresholds
        """
        self.model = model
        self.n_splits = n_splits
        self.scoring = scoring
        self.group_column = group_column
        self.random_state = random_state
        self.estimator_type = None
        self.result = None
        self.cv_strategy = None
        self.__large_limit = 20000

        # TODO: Make user able to choose cv_strategy if it is not selected use autocv mechansim
        # TODO: Make user able to select desired scorings with simple strings.

    @property
    def large_limit(self):
        return self.__large_limit

    @large_limit.setter
    def large_limit(self, value):
        self.__large_limit = value


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
        if self.n_splits is None:
            # Set default number of splits based on dataset size and estimator type
            if size <= 2500 or size > self.__large_limit:
                self.n_splits = 10
            elif size > 10000 and self.estimator_type == 'nn':
                self.n_splits = 3
            else:
                self.n_splits = 5
        elif self.n_splits > size:
            raise ValueError(f"Number of splits (cv={self.n_splits}) cannot be more than the number of data points (n={size}).")


    def cross_validate(self, X, y, force=False, n_jobs=None):
        """
        Perform automatic cross-validation.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            The input data.
        y: array-like, shape (n_samples,)
            The target variable.
        force: bool, optional (default=False)
            If True, forces cross-validation on large datasets even if it may be computationally expensive.
        n_jobs: int, optional (default=None)
            The number of jobs to run in parallel for cross-validation. None means 1 unless in a joblib.parallel_backend context.
            -1 means using all processors.

        Returns:
        float
            The mean of the cross-validation scores.
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

        # Use Group CV strategy if specified
        if self.group_column is not None:
            groups = self.group_column
            if problem_type == 'classification' and self._is_imbalanced(y):
                cv_strategy = StratifiedGroupKFold(n_splits=self.n_splits, random_state=self.random_state)
            else:
                cv_strategy = GroupKFold(n_splits=self.n_splits)
        else:
            groups = None

        # Set cv_strategy attribute
        self.cv_strategy = cv_strategy

        # Use default scoring if it is not set manually
        if self.scoring is None:
            self.scoring = set_default_scoring(y)

        # Perform cross-validation
        results = sklearn_cv(self.model, X, y, cv=cv_strategy, scoring=self.scoring, groups=groups, n_jobs=n_jobs)
                
        # Separate the fit_time and score_time from the results
        fit_time = results.pop('fit_time')
        score_time = results.pop('score_time')

        # The remaining items in results are the scores
        scores = results

        # Calculate the average scores
        average_scores = {f'average_{key}': np.mean(value) for key, value in scores.items()}

        # Create the Result object
        self.result = Result(
            fit_time=fit_time,
            score_time=score_time,
            average_fit_time=np.mean(fit_time),
            average_score_time=np.mean(score_time),
            scores=scores,
            average_scores=average_scores
        )

        # Return the average scores
        return average_scores


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
        if size <= 150:
            return LeavePOut(p=2)
        elif size <= 1000:
            return LeaveOneOut()
        elif problem_type == 'classification':
            y = self._encode_labels_if_needed(y)
            if self._is_imbalanced(y):
                if size < self.__large_limit:
                    return StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state)
                else:
                    return StratifiedShuffleSplit(n_splits=self.n_splits, random_state=self.random_state)
            else:
                if size < self.__large_limit:
                    return KFold(n_splits=self.n_splits)
                else:
                    return ShuffleSplit(n_splits=self.n_splits)
        else:
            if size < self.__large_limit:
                return KFold(n_splits=self.n_splits)
            else:
                return ShuffleSplit(n_splits=self.n_splits)


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

    def summary(self): # TODO: Show min max accuracy for each scoring and make sumary more appealing
        """
        Print a summary of the cross-validation results.
        """
        if self.result is None:
            print("No results available. Please run cross-validation first.")
            return

        print("Cross-Validation Summary:")
        print("-------------------------")
        print(f"Model: {self.model}")
        print(f"Cross-Validation Strategy: {self.cv_strategy}")
        print(f"Average Fit Time: {self.result.average_fit_time:.4f} seconds")
        print(f"Average Score Time: {self.result.average_score_time:.4f} seconds")
        print("Scores:")
        for key, value in self.result.average_scores.items():
            print(f"  {key}: {value:.4f}")

    def __repr__(self) -> str:
        model_name = self.model.__class__.__name__
        n_splits_info = f"n_splits={self.n_splits}" if self.n_splits is not None else "n_splits=auto"
        scoring_info = f"scoring={self.scoring}" if self.scoring is not None else "scoring=default"
        group_info = f"group_column={bool(self.group_column)}" if self.group_column is not None else "group_column=None"
        cv_strategy_name = self.cv_strategy.__class__.__name__ if self.cv_strategy else "cv_strategy=auto"
        
        return f"AutoCV(model={model_name}, {n_splits_info}, {scoring_info}, {group_info}, cv_strategy={cv_strategy_name})"
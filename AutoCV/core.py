"""
Core module for the AutoCV.
"""

import numpy as np
from utils import check_scoring
from sklearn.model_selection import cross_val_score

class AutoCV():
    def __init__(self) -> None:
        pass

def cross_validate(
    estimator,
    X,
    y=None,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
):
    # To ensure multimetric format is not supported
    scorer = check_scoring(estimator, scoring=scoring)

    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring={"score": scorer},
        cv=cv,
        n_jobs=n_jobs,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch
)
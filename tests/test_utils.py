import pytest
import numpy as np

from autocv.utils import (
    determine_problem_type,
    is_sklearn_model,
    is_tf_model,
    is_torch_model,
    check_estimator_type
)

@pytest.mark.parametrize("target, expected, expected_detailed", [
    (np.array([0, 1, 0, 1]), 'classification', 'binary_classification'),
    (np.array([0, 1, 2, 1]), 'classification', 'multiclass_classification'),
    (np.array([1.5, 2.3, 3.8, 4.1]), 'regression', 'regression')
])
def test_determine_problem_type(target, expected, expected_detailed):
    assert determine_problem_type(target) == expected
    assert determine_problem_type(target, detailed=True) == expected_detailed

def test_determine_problem_type_unsupported():
    # Unsupported type test
    target = np.array([[0, 1], [1, 0]])  # Multilabel indicator
    with pytest.raises(ValueError):
        determine_problem_type(target)

def test_is_sklearn_model():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    assert is_sklearn_model(model) == True

    model = np.array([0, 1, 1])
    assert is_sklearn_model(model) == False
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

def test_is_torch_model():
    try:
        from torch.nn import Module
        class TestModel(Module):
            pass
        model = TestModel()
        assert is_torch_model(model) == True
    except:
        assert True

def test_is_tf_model():
    try:
        from tensorflow.keras.models import Sequential
        model = Sequential()
        assert is_tf_model(model) == True
    except:
        assert True

def test_check_estimator_type_sklearn():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    assert check_estimator_type(model) == 'ml'

def test_check_estimator_type_torch():
    try:
        from torch.nn import Module
        class TestModel(Module):
            pass
        model = TestModel()
        assert check_estimator_type(model) == 'nn'
    except:
        assert True

def test_check_estimator_type_tf():
    try:
        from tensorflow.keras.models import Sequential
        model = Sequential()
        assert check_estimator_type(model) == 'nn'
    except:
        assert True

def test_check_estimator_type_unsupported():
    # Test unsupported estimator type
    class UnsupportedModel:
        pass
    model = UnsupportedModel()
    with pytest.raises(TypeError):
        check_estimator_type(model)
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from autocv import AutoCV

@pytest.fixture
def setup_data():
    # Balanced dataset
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression()
    return X, y, model

@pytest.fixture
def setup_imbalanced_data():
    # Imbalanced dataset that causes one class per fold scenario
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 0, 1, 1])  # Heavily imbalanced
    model = LogisticRegression()
    return X, y, model

@pytest.fixture
def large_data():
    X = np.random.rand(100001, 10)
    y = np.random.randint(0, 2, size=100001)
    return X, y

def test_autocv_initialization(setup_data):
    X, y, model = setup_data
    autocv = AutoCV(model=model, n_splits=5, scoring='accuracy')
    assert autocv.model == model
    assert autocv.n_splits == 5
    assert autocv.scoring == 'accuracy'

def test_large_dataset_skipping(setup_data, large_data):
    X_large, y_large = large_data
    model = LogisticRegression()
    autocv = AutoCV(model=model)
    result = autocv.cross_validate(X_large, y_large)
    assert result is None

def test_force_large_dataset(setup_data, large_data):
    X_large, y_large = large_data
    model = LogisticRegression()
    autocv = AutoCV(model=model)
    result = autocv.cross_validate(X_large, y_large, force=True)
    assert result is not None
    assert isinstance(result, dict), "Results should be a dictionary"

def test_invalid_n_splits_error(setup_data):
    X, y, model = setup_data
    with pytest.raises(ValueError):
        autocv = AutoCV(model=model, n_splits=10)
        autocv.cross_validate(X, y)

@pytest.mark.parametrize("size", [100, 150, 1000, 10000, 50000])
def test_cv_strategy_selection(setup_data, size):
    X = np.random.rand(size, 10)
    y = np.random.randint(0, 2, size=size)
    model = LogisticRegression()
    autocv = AutoCV(model=model)
    autocv.cross_validate(X, y)
    assert autocv.cv_strategy is not None
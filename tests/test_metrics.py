import pytest
from unittest.mock import patch
from autocv.metrics import (
    set_default_scoring,
    binary_classification_scorers,
    multi_class_classification_scorers,
    regression_scorers,
    determine_problem_type
)

# Test data
binary_target = [0, 1, 0, 1]
multi_class_target = [0, 1, 2, 1, 0]
regression_target = [2.5, 0.5, 1.3, 3.8, 2.2]

# Mock function to replace determine_problem_type
def mock_determine_problem_type(target, detailed=True):
    if target == binary_target:
        return 'binary_classification'
    elif target == multi_class_target:
        return 'multiclass_classification'
    elif target == regression_target:
        return 'regression'
    else:
        return 'unknown'

@patch('autocv.utils.determine_problem_type', side_effect=mock_determine_problem_type)
def test_set_default_scoring(mock_determine_problem_type):
    # Test binary classification
    result = set_default_scoring(binary_target)
    assert result == binary_classification_scorers, f"Expected {binary_classification_scorers}, but got {result}"

    # Test multi-class classification
    result = set_default_scoring(multi_class_target)
    assert result == multi_class_classification_scorers, f"Expected {multi_class_classification_scorers}, but got {result}"

    # Test regression
    result = set_default_scoring(regression_target)
    assert result == regression_scorers, f"Expected {regression_scorers}, but got {result}"

    # Test unsupported problem type
    with pytest.raises(ValueError, match="Unsupported or unrecognized target type: unknown"):
        set_default_scoring([None, None, None, None, None])
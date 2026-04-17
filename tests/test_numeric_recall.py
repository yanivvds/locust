import pandas as pd
from evaluation.metrics import numeric_recall


def test_exact_match():
    gt = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    pred = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert numeric_recall(gt, pred) == 1.0


def test_no_match():
    gt = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    pred = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    assert numeric_recall(gt, pred) == 0.0


def test_partial_match():
    gt = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    pred = pd.DataFrame({'A': [1, 3], 'B': [9, 9]})
    assert numeric_recall(gt, pred) == 0.5


def test_ignores_row_structure():
    """Values swapped across rows still count as matched."""
    gt = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    pred = pd.DataFrame({'A': [2, 1], 'B': [4, 3]})
    assert numeric_recall(gt, pred) == 1.0


def test_ignores_strings():
    gt = pd.DataFrame({'A': ['foo', 1], 'B': ['bar', 2]})
    pred = pd.DataFrame({'X': ['baz', 1], 'Y': ['qux', 2]})
    assert numeric_recall(gt, pred) == 1.0


def test_transposed_format():
    """Wide vs long format produces the same numeric bag."""
    gt = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    pred = pd.DataFrame({'X': [1, 4], 'Y': [2, 5], 'Z': [3, 6]})
    assert numeric_recall(gt, pred) == 1.0


def test_nan_ignored():
    gt = pd.DataFrame({'A': [1, float('nan')], 'B': [3, 4]})
    pred = pd.DataFrame({'A': [1, 3, 4]})
    assert numeric_recall(gt, pred) == 1.0


def test_duplicate_values():
    gt = pd.DataFrame({'A': [1, 1, 2]})
    pred = pd.DataFrame({'A': [1, 2, 3]})
    assert numeric_recall(gt, pred) == 2 / 3


def test_empty_ground_truth():
    assert numeric_recall(pd.DataFrame(), pd.DataFrame({'A': [1]})) == 0.0
    assert numeric_recall(pd.DataFrame(), pd.DataFrame()) == 1.0


def test_empty_predicted():
    gt = pd.DataFrame({'A': [1, 2]})
    assert numeric_recall(gt, pd.DataFrame()) == 0.0


def test_only_strings_in_gt():
    gt = pd.DataFrame({'A': ['foo', 'bar']})
    pred = pd.DataFrame({'A': ['foo', 'bar']})
    assert numeric_recall(gt, pred) == 1.0


def test_only_strings_in_gt_mismatch():
    gt = pd.DataFrame({'A': ['foo', 'bar']})
    pred = pd.DataFrame({'A': [1, 2]})
    assert numeric_recall(gt, pred) == 0.0

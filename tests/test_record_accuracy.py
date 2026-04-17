import pandas as pd
from evaluation.metrics import record_accuracy


def test_exact_match():
    gt = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    pred = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert record_accuracy(gt, pred) == 1.0


def test_no_match():
    gt = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    pred = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    assert record_accuracy(gt, pred) == 0.0


def test_partial_match():
    gt = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    pred = pd.DataFrame({'A': [1], 'B': [4]})
    assert abs(record_accuracy(gt, pred) - 1 / 3) < 1e-9


def test_column_name_agnostic():
    gt = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    pred = pd.DataFrame({'X': [1, 2], 'Y': [3, 4]})
    assert record_accuracy(gt, pred) == 1.0


def test_row_order_agnostic():
    gt = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    pred = pd.DataFrame({'A': [2, 1], 'B': [4, 3]})
    assert record_accuracy(gt, pred) == 1.0


def test_nan_filtered():
    gt = pd.DataFrame({'A': [1, float('nan')], 'B': [3, 4]})
    pred = pd.DataFrame({'A': [1, 4], 'B': [3, float('nan')]})
    assert record_accuracy(gt, pred) == 1.0


def test_transposed_gt():
    gt = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    pred = pd.DataFrame({'X': [1, 4], 'Y': [2, 5], 'Z': [3, 6]})  # gt.T
    assert record_accuracy(gt, pred) == 1.0


def test_square_df_no_transpose():
    """Square DataFrames skip the transpose check to avoid ambiguity."""
    gt = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    pred = pd.DataFrame({'X': [1, 3], 'Y': [2, 4]})  # gt.T
    assert record_accuracy(gt, pred) == 0.0


def test_empty_ground_truth():
    gt = pd.DataFrame()
    pred = pd.DataFrame({'A': [1]})
    assert record_accuracy(gt, pred) == 0.0

    assert record_accuracy(pd.DataFrame(), pd.DataFrame()) == 1.0


def test_empty_predicted():
    gt = pd.DataFrame({'A': [1, 2]})
    pred = pd.DataFrame()
    assert record_accuracy(gt, pred) == 0.0


def test_duplicate_rows():
    gt = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 3, 4]})
    pred = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert abs(record_accuracy(gt, pred) - 2 / 3) < 1e-9

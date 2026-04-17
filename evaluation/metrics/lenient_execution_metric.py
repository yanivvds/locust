"""
    Lenient execution accuracy metric. Complements the strict execution accuracy by
    ignoring structural differences (column names, shape, pivot orientation) and
    comparing only the multiset of cell values.
"""
import pandas as pd
from collections import Counter


def cell_multiset(df: pd.DataFrame, numeric_precision: int) -> Counter:
    """
        Flatten a DataFrame into a multiset of its non-null cell values.
        Numeric values are rounded to `numeric_precision` decimals; strings are kept as-is.
    """
    values = []
    for _, row in df.iterrows():
        for v in row:
            if pd.isna(v):
                continue
            if isinstance(v, (int, float)):
                values.append(('num', round(float(v), numeric_precision)))
            else:
                values.append(('str', str(v)))
    return Counter(values)


def lenient_execution_accuracy(ground_truth: pd.DataFrame,
                               predicted: pd.DataFrame,
                               numeric_precision: int = 5) -> float:
    """
        Lenient execution accuracy: 1.0 if the multiset of non-null cell values matches
        exactly, else 0.0.  Column names, row/column ordering, DataFrame shape, and
        pivot orientation are all ignored.

        :param ground_truth: ground-truth DataFrame
        :param predicted: predicted DataFrame
        :param numeric_precision: rounding tolerance for float comparisons
        :returns: 1.0 if leniently equivalent, 0.0 otherwise
    """
    if len(ground_truth) == 0:
        return 1.0 if len(predicted) == 0 else 0.0
    if len(predicted) == 0:
        return 0.0
    return 1.0 if cell_multiset(ground_truth, numeric_precision) == cell_multiset(predicted, numeric_precision) else 0.0

import pandas as pd
from collections import Counter


def _numeric_multiset(df: pd.DataFrame, numeric_precision: int) -> Counter:
    """Extracts all numeric values from a DataFrame into a flat multiset (Counter),
    ignoring string values, column names, row structure, and NaN."""
    values = []
    for _, row in df.iterrows():
        for v in row:
            if isinstance(v, (int, float)) and not pd.isna(v):
                values.append(round(float(v), numeric_precision))
    return Counter(values)


def numeric_recall(ground_truth: pd.DataFrame, predicted: pd.DataFrame, numeric_precision: int = 5) -> float:
    """
        Format-independent numeric recall.

        Compares the multiset of all numeric values across both DataFrames,
        ignoring column names, row structure, and string values. This is a
        lenient metric that complements the stricter row-aware record_accuracy.

        :param ground_truth: ground truth dataframe
        :param predicted: predicted dataframe
        :param numeric_precision: rounding tolerance for float comparisons
        :returns: fraction of GT numeric values found in the prediction (0 to 1)
    """
    if len(ground_truth) == 0:
        return 1.0 if len(predicted) == 0 else 0.0

    if len(predicted) == 0:
        return 0.0

    gt_values = _numeric_multiset(ground_truth, numeric_precision)
    pred_values = _numeric_multiset(predicted, numeric_precision)

    if not gt_values:
        return 1.0 if not pred_values else 0.0

    correct = sum(min(gt_values[v], pred_values[v]) for v in gt_values)
    return correct / sum(gt_values.values())

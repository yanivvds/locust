import pandas as pd
from collections import Counter


def get_canonical_rows(df: pd.DataFrame, numeric_precision: int) -> list:
    """Converts each row of a DataFrame into a hashable representation of its value multiset."""
    canonical_rows = []
    for _, row in df.iterrows():
        values = []
        for v in row:
            if pd.isna(v):
                continue
            elif isinstance(v, (int, float)):
                values.append(round(float(v), numeric_precision))
            else:
                values.append(v)
        # Create a hashable representation of the multiset of values in the row
        canonical_rows.append(frozenset(Counter(values).items()))
    return canonical_rows


def record_accuracy(ground_truth: pd.DataFrame, predicted: pd.DataFrame, numeric_precision: int = 5) -> float:
    """
        Calculates record accuracy by comparing multisets of cell values row by row,
        disregarding column names and the order of values within a row.
        A predicted row is considered equal if a row with the same multiset
        of cell values exists in the ground-truth dataframe.

        :param ground_truth: ground truth dataframe
        :param predicted: predicted dataframe
        :param numeric_precision: rounding tolerance for float comparisons
        :returns: accuracy between 0 and 1 for the number of matching rows divided by total rows in the ground truth DF
    """
    if len(ground_truth) == 0:
        return 1.0 if len(predicted) == 0 else 0.0

    if len(predicted) == 0:
        return 0.0

    gt_canonical_rows = get_canonical_rows(ground_truth, numeric_precision)
    pred_canonical_rows = get_canonical_rows(predicted, numeric_precision)

    # Count the occurrences of each unique row multiset in the predicted data
    pred_counts = Counter(pred_canonical_rows)

    num_correct_records = 0
    # Iterate through each ground truth row and see if a match can be found in the predicted rows
    for gt_row in gt_canonical_rows:
        if pred_counts[gt_row] > 0:
            num_correct_records += 1
            # Decrement the count for that predicted row, "consuming" the match
            pred_counts[gt_row] -= 1

    num_total_records = len(ground_truth)
    score = num_correct_records / num_total_records

    # If the ground truth might be transposed (e.g. PIVOT vs UNPIVOT), check
    # whether comparing against the transposed GT yields a better score.
    if score < 1.0 and ground_truth.shape[0] != ground_truth.shape[1]:
        gt_transposed = ground_truth.T.reset_index(drop=True)
        gt_t_canonical = get_canonical_rows(gt_transposed, numeric_precision)
        pred_counts_t = Counter(pred_canonical_rows)
        correct_t = 0
        for gt_row in gt_t_canonical:
            if pred_counts_t[gt_row] > 0:
                correct_t += 1
                pred_counts_t[gt_row] -= 1
        score_t = correct_t / len(gt_transposed)
        score = max(score, score_t)

    return score

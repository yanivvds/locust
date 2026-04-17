"""
This module contains the Component Match metric for the query generation task.
This is metric only relevant when generating SQL queries.
"""
import sqlglot
from typing import Set, Tuple, Dict, Union


def get_select_components(expression: sqlglot.exp.Expression) -> Set[Tuple]:
    """Extracts components from the SELECT clause."""
    components = {}
    for select_element in expression.find_all(sqlglot.exp.Select):
        for projection in select_element.expressions:
            if isinstance(projection, sqlglot.exp.Star):
                if '*' not in components:
                    components['*'] = set()
                continue

            if isinstance(projection, sqlglot.exp.Alias):
                projection = projection.this

            agg = str(projection.key).lower() if isinstance(projection, sqlglot.exp.AggFunc) else None

            col_expr = projection
            if agg:
                col_expr = projection.this

            if hasattr(col_expr, 'this') and col_expr.this is not None:
                col = str(col_expr.this)
            else:
                col = str(col_expr)

            if col not in components:
                components[col] = set()
            if agg:
                components[col].add(agg)

    return {tuple(sorted(aggs)) + (col,) for col, aggs in components.items()}


def get_where_components(expression: sqlglot.exp.Expression) -> Set[Tuple]:
    """Extracts components from the WHERE clause."""
    components = set()
    for where in expression.find_all(sqlglot.exp.Where):
        for condition in where.find_all(sqlglot.exp.Binary):
            if isinstance(condition.left, sqlglot.exp.Column) and isinstance(condition.right, sqlglot.exp.Literal):
                op = condition.__class__.__name__.lower()
                components.add((str(condition.left.this), op, condition.right.this))
    return components


def get_groupby_components(expression: sqlglot.exp.Expression) -> Set[str]:
    """Extracts components from the GROUP BY clause."""
    components = set()
    for group_by in expression.find_all(sqlglot.exp.Group):
        for col in group_by.expressions:
            components.add(str(col.this))
    return components


def get_orderby_components(expression: sqlglot.exp.Expression) -> Set[Tuple]:
    """Extracts components from the ORDER BY clause."""
    components = set()
    for order_by in expression.find_all(sqlglot.exp.Order):
        for ordered_item in order_by.expressions:
            direction = ordered_item.args.get('kind', 'asc').lower()
            components.add((str(ordered_item.this.this), direction))
    return components


SINGLE_ROW_EQUIVALENT_AGGS = {'sum', 'avg', 'min', 'max'}
def _pivot_groups_are_singletons(pivot: sqlglot.exp.Pivot) -> bool:
    """
        True when each PIVOT group can contain at most one source row:
        every dimension in the `FOR … IN (…)` clause has exactly one
        literal value. In the CBS OData3 layout (one row per dimension tuple)
        this guarantees single-row groups, so SUM/AVG/MIN/MAX all coincide.
    """
    fields = pivot.args.get("fields", [])
    if not fields:
        return False
    for field in fields:
        if not isinstance(field, sqlglot.exp.In):
            return False
        n_literals = sum(1 for v in field.expressions if isinstance(v, sqlglot.exp.Literal))
        if n_literals != 1:
            return False
    return True


def get_pivot_components(expression: sqlglot.exp.Expression) -> Set[Tuple]:
    """Extracts components from the PIVOT clause."""
    components = set()
    pivot = expression.find(sqlglot.exp.Pivot)
    if not pivot:
        return components

    agg_func = pivot.expressions[0]
    agg_name = str(agg_func.key).lower()

    # Treat equivalent aggregations as a canonical 'identity' operator
    # whenever each PIVOT group provably contains at most one row.
    if agg_name in SINGLE_ROW_EQUIVALENT_AGGS and _pivot_groups_are_singletons(pivot):
        agg_name = "identity"

    if not isinstance(agg_func.this, sqlglot.exp.Column):
        agg_col = agg_func.this
    else:
        agg_col = str(agg_func.this.this)

    for field in pivot.args.get("fields", []):
        if isinstance(field, sqlglot.exp.In):
            pivot_col = str(field.this.this)
            pivot_values = frozenset({val.this for val in field.expressions})
            components.add((agg_name, agg_col, pivot_col, pivot_values))

    return components


def calculate_f1(predicted_set: Set, truth_set: Set) -> Dict[str, float]:
    """Calculates Precision, Recall, and F1 score for two sets."""
    if not truth_set and not predicted_set:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if not truth_set or not predicted_set:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    tp = len(predicted_set.intersection(truth_set))
    fp = len(predicted_set.difference(truth_set))
    fn = len(truth_set.difference(predicted_set))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def calculate_component_matching(predicted_sql: str, ground_truth_sql: str) -> Dict[str, Union[Dict[str, float], float]]:
    """Calculates the component matching scores for SQL queries."""
    try:
        predicted_ast = sqlglot.parse_one(predicted_sql, read='duckdb')
        truth_ast = sqlglot.parse_one(ground_truth_sql, read='duckdb')
    except Exception:
        return {comp + '_f1': 0.0 for comp in ['select', 'where', 'groupby', 'orderby', 'pivot']}

    scores = {}

    pred_select = get_select_components(predicted_ast)
    truth_select = get_select_components(truth_ast)
    scores['select_f1'] = calculate_f1(pred_select, truth_select)['f1']

    pred_where = get_where_components(predicted_ast)
    truth_where = get_where_components(truth_ast)
    scores['where_f1'] = calculate_f1(pred_where, truth_where)['f1']

    pred_groupby = get_groupby_components(predicted_ast)
    truth_groupby = get_groupby_components(truth_ast)
    scores['groupby_f1'] = calculate_f1(pred_groupby, truth_groupby)['f1']

    pred_orderby = get_orderby_components(predicted_ast)
    truth_orderby = get_orderby_components(truth_ast)
    scores['orderby_f1'] = calculate_f1(pred_orderby, truth_orderby)['f1']

    pred_pivot = get_pivot_components(predicted_ast)
    truth_pivot = get_pivot_components(truth_ast)
    scores['pivot_f1'] = calculate_f1(pred_pivot, truth_pivot)['f1']

    return scores

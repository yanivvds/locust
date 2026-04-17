from __future__ import annotations

import os
import pandas as pd
import sqlglot
from beartype import beartype
from typing import Tuple, List

import config
from pipeline.db_executor import DBExecutor
from s_expression import Expression, ParsedExpression
from s_expression.operators import Value


class Prop(Expression):
    """
        The PROP expression calculates the percentage (proportion) of a given selector dimension over the
        sum of the entire dimension group. For example, `(PROP (DIM Perioden (2021JJ00)) (VALUE ...))`
        would sum all the extracted measure values from the VALUE expressions over the dimension
        `Perioden`, and then calculate what percentage of the total is in `2021JJ00`.

        :param sexp: the parsed S-expression, of which the evaluation instantiates this object
        :param selector: axes to aggregate over (dim group) and code to calculate percentage for
        :param sub_expression: sub-expression that will be executed and used for aggregating over
    """
    @beartype
    def __init__(self, sexp: ParsedExpression, selector: Tuple[str, List[str]], sub_expression: Expression):
        super().__init__(sexp)
        self.sub_expression = sub_expression
        self.selector = selector

    def __call__(self,
                 sql: bool = False,
                 odata4: bool = False,
                 simplified: bool = False,
                 offline: bool = False,
                 verbose: bool = False) -> Tuple[Prop, pd.DataFrame]:
        if sql:
            sub_exp = self.sub_expression
            while 1:  # Get inner VALUE expression
                if isinstance(sub_exp, Value):
                    break
                sub_exp = sub_exp.sub_expression

            if simplified:
                query = self.odata3_sql_simplified
            else:
                query = self.odata3_sql

            db = DBExecutor(tables=[sub_exp.table], measures=sub_exp.measures, dims=sub_exp.dimensions,
                            operator_name=self._operator)
            answer, code_labels, _ = db.query_db(query=query, simplified=simplified)
            self.mapper = code_labels
        else:
            sub_exp, table = self.sub_expression(offline=offline, odata4=odata4, verbose=verbose)

            self.mapper = sub_exp.mapper
            self.intermediate_results = [table] + sub_exp.intermediate_results
            totals = self._aggregate(table, [self.selector[0]], pd.DataFrame.sum, {})

            answer = totals.T
            selector_labels = tuple(self.mapper[c] for c in self.selector[1])
            selector_vals = table.filter(like=selector_labels[0]).T
            selector_vals.index = answer.index

            answer[selector_labels + tuple('' for _ in range(answer.columns.nlevels-1))] = selector_vals
            answer[('%',) + tuple('' for _ in range(answer.columns.nlevels-1))] = 100 / totals.T * selector_vals

        if verbose:
            self._print_answer(answer)

        return self, answer

    @property
    def odata3_sql(self):
        """
            === Example PROP s-expression as OData3 SQL ===
            SELECT CONCAT_WS(', ', BestemmingEnSeizoen, Marges, Measure, Vakantiekenmerken) AS Dimension_Measure,
                   SUM(Value) AS 'SUM[Perioden]',
                   SUM(CASE WHEN Perioden IN ('2021JJ00') THEN Value ELSE 0 END) AS "2021JJ00",
                   ROUND(
                       SUM(CASE WHEN Perioden IN ('2021JJ00') THEN Value ELSE 0 END) * 100.0 / SUM(Value),
                       2
                   ) AS "%"
            FROM (
                SELECT Measure, Value, Marges, Perioden, BestemmingEnSeizoen, Vakantiekenmerken
                FROM '<parquet_file>'
                UNPIVOT(Value FOR Measure IN (D004645))
            )
            WHERE Marges IN ('MW00000')
                AND Perioden IN ('2021JJ00', '2022JJ00')
                AND BestemmingEnSeizoen IN ('L008691', 'L999996')
                AND Vakantiekenmerken IN ('T001460')
            GROUP BY Dimension_Measure
        """
        sub_exp = self.sub_expression
        while not isinstance(sub_exp, Value):
            sub_exp = sub_exp.sub_expression

        group_by_cols = sorted([str(d[0]) for d in sub_exp.dimensions if str(d[0]) != self.selector[0]] + ['Measure'])
        selector_values = ", ".join(f"'{c}'" for c in self.selector[1])
        
        where_parts = []
        for dim, codes in sub_exp.dimensions:
            if len(codes) > 0:
                cod_str = ", ".join(f"'{c}'" for c in codes)
                where_parts.append(f"{dim} IN ({cod_str})")
        where_clauses = " AND ".join(where_parts)

        sql = f"""
            SELECT CONCAT_WS(', ', {', '.join(group_by_cols)}) AS Dimension_Measure,
                   SUM(Value) AS 'SUM[{self.selector[0]}]',
                   SUM(CASE WHEN {self.selector[0]} IN ({selector_values}) THEN Value ELSE 0 END) AS \"{"_".join(self.selector[1])}\",
                   ROUND(
                       SUM(CASE WHEN {self.selector[0]} IN ({selector_values}) THEN Value ELSE 0 END) * 100.0 / SUM(Value), 2
                   ) AS "%"
            FROM (
                SELECT Measure, Value, {', '.join(str(g) for g, _ in sub_exp.dimensions)}
                FROM '{os.path.relpath(config.DB_ODATA3_FILES)}/{sub_exp.table}.parquet'
                UNPIVOT (
                    Value FOR Measure IN ("{'", "'.join(str(msr) for msr in sub_exp.measures)}")
                )
            )
            {'WHERE ' + where_clauses if where_clauses else ''}
            GROUP BY Dimension_Measure;
        """
        return sqlglot.parse_one(sql).sql(pretty=True)

    @property
    def odata3_sql_simplified(self):
        """There is no simplified version for the PROP type query. Defaults to the OData3 SQL implementation"""
        return self.odata3_sql


if __name__ == '__main__':
    from s_expression.parser import parse, eval

    # Multi dimension table example
    s, a = eval(parse("""
    (PROP (DIM WijkenEnBuurten (WK037501)) (VALUE 84517NED (MSR (M002463 < 284)) (DIM WijkenEnBuurten ())))
    """), sql=True, simplified=True, verbose=True)

    # Single dimension table example
    eval(parse("""
        (PROP
        (DIM Perioden (2007JJ00))
        (VALUE 37789ksz
            (MSR (D000292))
            (DIM Perioden (2007JJ00 2016JJ00 2008JJ00 2013JJ00))
        )
    )
    """), sql=True, verbose=True)

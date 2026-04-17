from __future__ import annotations

import pandas as pd
import sqlglot
from abc import ABC
from beartype import beartype
from typing import List, Tuple, Set, FrozenSet

from pipeline.db_executor import DBExecutor
from pipeline.odata_executor import CodeLabelMapper
from s_expression import Expression, ParsedExpression, Table, Measure, Dimension
from s_expression.operators import Value


class SimpleAggregator(Expression, ABC):
    """
        Shared logic for simple aggregator functions SUM, AVG, MIN, MAX

        :param sexp: the parsed S-expression, of which the evaluation instantiates this object
        :param selectors: axes to aggregate the intermediate pivot table results over
        :param sub_expression: sub-expression that will be executed and used for aggregating over
    """
    @beartype
    def __init__(self, sexp: ParsedExpression, selectors: List[str], sub_expression: Expression):
        super().__init__(sexp)
        self.sub_expression = sub_expression
        self.selectors = selectors

        # Selectors of complex inner queries must match
        sub_exps = [self.sub_expression]
        for sub_exp in sub_exps:  # Get inner VALUE expression(s)
            if isinstance(sub_exp, Value):
                continue
            else:
                # TODO: implement aggregations over complex inner queries for non-measure aggregations
                if len(self.selectors):
                    raise NotImplementedError("Can't execute aggregations over complex inner queries "
                                              "with a dimension as the aggregation selector yet.")

                inner_selectors = [] if 'Measure' in sub_exp.selectors else sub_exp.selectors
                if inner_selectors != self.selectors:
                    raise ValueError(f"Selectors of all complex inner expressions must be identical.\n"
                                     f"Selector {self.selectors} in the {self._operator} expression\n"
                                     f"does not match {sub_exp.selectors} in the {sub_exp._operator} sub-expression")

            if hasattr(sub_exp, 'sub_expressions'):
                sub_exps.extend(sub_exp.sub_expressions)
            else:
                sub_exps.append(sub_exp.sub_expression)

    def _get_sub_exp_filters(self) -> Tuple[List[Value], Set[Table], Set[Measure], Set[Tuple[Dimension, FrozenSet[Dimension]]]]:
        """
            Helper function for getting all tables, measures and dimensions from all
            inner VALUE expressions in this S-expression, regardless of their depth.

            :returns: Tuple containing a list of inner VALUE expressions, set with tables,
                      set with measures and set with dimensions
        """
        tables = set()
        measures = set()
        dimensions = set()
        value_exps = []
        sub_exps = [self.sub_expression]
        for sub_exp in sub_exps:  # Get inner VALUE expression(s)
            if isinstance(sub_exp, Value):
                tables |= { sub_exp.table }
                measures |= sub_exp.measures
                dimensions |= sub_exp.dimensions
                value_exps.append(sub_exp)
                continue

            if hasattr(sub_exp, 'sub_expressions'):
                sub_exps.extend(sub_exp.sub_expressions)
            else:
                sub_exps.append(sub_exp.sub_expression)

        return value_exps, tables, measures, dimensions

    def _execute_sql(self, simplified: bool = False) -> Tuple[pd.DataFrame, CodeLabelMapper]:
        """
            Execute and return the result from the SQL query corresponding with this S-expression.

            :param simplified: bool to indicate to use the simplified OData3 query
            :return: resulting DataFrame and code-to-label mapping dictionary
        """
        _, tables, measures, dimensions = self._get_sub_exp_filters()

        if simplified:
            query = self.odata3_sql_simplified
        else:
            query = self.odata3_sql

        db = DBExecutor(tables=list(tables), measures=measures, dims=dimensions, operator_name=self._operator)
        answer, code_labels, _ = db.query_db(query=query, simplified=simplified)
        return answer, code_labels

    @property
    def odata3_sql(self):
        """
            === Example SUM s-expression as OData3 SQL ===
            SELECT *
            FROM (
                SELECT Measure, Value, Perioden, BestemmingEnSeizoen, Marges, Vakantiekenmerken
                FROM '<parquet_file>'
                UNPIVOT (
                    Value FOR Measure IN (D004645)
                )
            PIVOT (
                SUM(Value)
                FOR BestemmingEnSeizoen IN ('L008691', 'L999996')
                    Marges IN ('MW00000')
                    Vakantiekenmerken IN ('T001460')
                    Perioden IN ('2021JJ00', '2022JJ00'))
                GROUP BY Measure
            );
        """
        # Get inner select from sub expression to fetch data to aggregate
        sub_sql = sqlglot.parse_one(self.sub_expression.odata3_sql)

        if not isinstance(self.sub_expression, Value):  # Aggregate over complex inner query (e.g. AGGJOIN)
            if len(self.selectors) == 0:
                sql = f"""
                    SELECT Measure, {self._operator}(Value) AS {self._operator}
                    FROM ({sub_sql.sql()})
                    UNPIVOT (Value FOR name IN (COLUMNS(c -> NOT contains(c, 'Measure'))))
                    GROUP BY Measure
                """
                return sqlglot.parse_one(sql).sql(pretty=True)
            else:
                raise NotImplementedError("Can't build SQL aggregations over complex inner queries "
                                          "with a dimension as the aggregation selector yet.")

        select = sub_sql.find(sqlglot.exp.Subquery).find(sqlglot.exp.Select)

        # Copy existing PIVOT over from sub-expression
        for_statements = {}
        for node in sub_sql.find_all(sqlglot.exp.In):
            if 'Measure' not in node.sql():
                for_statements |= dict([node.sql().split(' IN ')])

        # Remove measure from the inner select if aggregating on measure
        if len(self.selectors) == 0:
            select.select(copy=False).set("expressions",
                                          [e for e in select.select().expressions if 'Measure' not in e.sql()])

        # Remove the selector from the existing FOR's in the PIVOT to add as WHERE
        for selector in self.selectors:
            if selector in for_statements:
                select.where(f"{selector} IN {for_statements[selector]}", append=True, copy=False)
            # If there is only ONE other FOR statement, PIVOT over the Measure
            if len(for_statements) <= 1:
                for_statements |= dict([node.sql().split(' IN ') for node in sub_sql.find_all(sqlglot.exp.In) if 'Measure' in node.sql()])
            for_statements.pop(selector, None)

        # Add a dummy column with the aggregated measures if applicable
        select_cols = '*'
        if not self.selectors:
            select_cols += ", '"
            select_cols += ';'.join([s.name for s in [e for e in select.find_all(sqlglot.exp.In)
                                                      if 'Measure' in e.sql()][0].expressions])
            select_cols += "' AS Measure"

        sql = f"""
            SELECT {select_cols}
            FROM ({select.sql()})
            PIVOT (
                {self._operator}(Value)
                FOR {'\n'.join('{} IN {}'.format(k, v) for k, v in for_statements.items())}
                {'GROUP BY Measure' if self.selectors else ''}
            )
        """
        return sqlglot.parse_one(sql).sql(pretty=True)

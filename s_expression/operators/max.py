from __future__ import annotations

import numpy as np
import pandas as pd
import sqlglot
from typing import Tuple

from odata_graph import engine
from s_expression.operators import Value
from s_expression.simple_aggregator import SimpleAggregator


class Max(SimpleAggregator):
    """
        Get the max values over given axes for the intermediate results of a given expression.
        When running as native S-expression, MAX works like a combined MAX and ARGMAX function.
        This means that always the context is returned from the initial tables. When running as
        SQL query, MAX works like a traditional MAX function over a PIVOT command.
    """
    def __call__(self,
                 sql: bool = False,
                 odata4: bool = False,
                 simplified: bool = False,
                 offline: bool = False,
                 verbose: bool = False) -> Tuple[Max, pd.DataFrame]:
        if sql:
            answer, code_labels = self._execute_sql(simplified=simplified)
            self.mapper = code_labels
        else:
            measure_units = {}
            if isinstance(self.sub_expression, Value) and len(self.selectors) == 0:
                measure_units = engine.validate_msr_unit_compatibility(self.sub_expression.measures)
            sub_exp, table = self.sub_expression(offline=offline, odata4=odata4, verbose=verbose)

            self.mapper = sub_exp.mapper
            measure_units = {self.mapper.get(k, k): v for k, v in measure_units.items()}

            max_values = self._aggregate(table, self.selectors, pd.DataFrame.max, measure_units)

            # TODO: single highest value assumption
            #  This method relies on that only ONE cell contains 1. If multiple columns have 1s
            #  (i.e. multiple equally highest values present in the table), it returns the first such column
            #  (by order in df.columns).
            if len(self.selectors) > 0:
                cols = [c if isinstance(table.columns, pd.MultiIndex) else (c,) for c in table.columns]
                selector_max = pd.DataFrame(  # Get the selector, being part of the corresponding column name, of each max value
                    np.array([
                        (pd.DataFrame(np.isclose(table, val), index=table.index, columns=cols) == 1).any().idxmax()
                        [table.columns.names.index(self.mapper.get(self.selectors[0], self.selectors[0]))]
                        for val in max_values.values.flatten()
                    ]).reshape(max_values.shape),
                    index=max_values.index,
                    columns=max_values.columns
                )

                # Combine the max dims from the selector with the corresponding values
                self.intermediate_results = [table] + sub_exp.intermediate_results
                answer = selector_max.compare(max_values).groupby(level=0, axis=1).apply(lambda dd: dd.agg(tuple, axis=1))
            else:
                answer = self._aggregate(table, self.selectors, pd.DataFrame.idxmax, measure_units)
                combiner = np.vectorize(lambda tup, nums: tup + (nums,), otypes=[object])
                answer[:] = combiner(answer.values, max_values.values)

        if verbose:
            self._print_answer(answer)

        return self, answer

    @property
    def odata3_sql_simplified(self) -> str:
        """
            === Example simplified MAX expression as OData3 SQL ===
            WITH RankedRows AS (
                SELECT RANK() OVER (ORDER BY Value DESC) as rnk,
                       Measure, Purpose, Period, Region,
                       Value AS 'MAX'
                FROM '<parquet-file>'
                UNPIVOT(Value FOR Measure IN ('BuildingPermitsAdditions_11', 'StockBalance_21'))
                WHERE Purpose IN ('T001419', 'A045372', 'A045375')
                AND Period IN ('2025KW01')
                AND Region IN ('NL01')
            )
            SELECT *
            FROM RankedRows
            WHERE rnk = 1;
        """
        sql = ""
        inner_sql = sqlglot.parse_one(self.sub_expression.odata3_sql_simplified)  # Get inner VALUE query
        _, _, measures, dimensions = self._get_sub_exp_filters()

        measure = str(list(measures)[0])
        selector = self.selectors[0] if len(self.selectors) > 0 else 'Value'
        sql += f"""
            SELECT 
                RANK() OVER (ORDER BY {'"' + measure + '"' if len(self.selectors) > 0 else 'Value'} DESC) as rnk,
                {'Measure' if len(self.selectors) == 0 else measure},
                {', '.join(str(group) for group, _ in dimensions if str(group) != selector) + ',' if len(dimensions) > 0 else ''}
                {selector} AS 'MAX[{selector}]'
        """

        # Add FROM clause from inner VALUE expression
        from_stmt = inner_sql.find(sqlglot.exp.From)
        if from_stmt:
            sql += from_stmt.sql()  + '\n'

        # Add UNPIVOT if required
        if len(self.selectors) == 0:
            sql += f"UNPIVOT(Value FOR Measure IN ('{"', '".join(map(str, measures))}'))\n"

        # Add WHERE cluse  from inner VALUE expression
        where_stmt = inner_sql.find(sqlglot.exp.Where)
        if where_stmt:
            sql += where_stmt.sql()

        outer_sql = f"""
            WITH RankedRows AS ({sql})
            SELECT *
            FROM RankedRows
            WHERE rnk = 1;
        """
        return sqlglot.parse_one(outer_sql).sql(pretty=True)


if __name__ == '__main__':
    from s_expression.parser import parse, eval

    eval(parse("""
        (MAX () (VALUE 85672NED (MSR (M000134)) (DIM BurgerlijkeStaat (1080)) (DIM CaribischNederland (GM9003 GM9001 GM9002)) (DIM Geslacht (T001038)) (DIM Leeftijd (10000)) (DIM Perioden (2021JJ00))))
    """), sql=True, verbose=True)

    eval(parse("""
        (MAX (BeroepsEnEigenVervoer) (VALUE 82836NED (MSR (M004375 D003778)) (DIM BeroepsEnEigenVervoer (100200 100100)) (DIM Vervoerstromen (A028222)) (DIM Perioden (2020JJ00))))
    """), verbose=True)

    # Max over dimension group and TWO measures
    eval(parse("""
        (MAX (Perioden) (VALUE 82836NED (MSR (M004375 D003778)) (DIM BeroepsEnEigenVervoer (100200 100100)) (DIM Vervoerstromen (A028222)) (DIM Perioden (2020JJ00 2021JJ00))))
    """), verbose=True)

    eval(parse("""(MAX
        (Perioden)
        (VALUE 85302NED 
            (MSR (D004645))
            (DIM Perioden (2021JJ00 2022JJ00))
            (DIM BestemmingEnSeizoen (L008691 L999996))
            (DIM Vakantiekenmerken (T001460))
            (DIM Marges (MW00000))
        )
    )
    """), verbose=True)

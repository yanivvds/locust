from __future__ import annotations

import os
import pandas as pd
import sqlglot
from typing import Tuple

import config
from odata_graph import engine
from s_expression.operators import Value
from s_expression.simple_aggregator import SimpleAggregator


class Sum(SimpleAggregator):
    """
        The SUM expression sums the resulting pivot tables from a sub-expression over a given column.
    """
    def __call__(self,
                 sql: bool = False,
                 odata4: bool = False,
                 simplified: bool = False,
                 offline: bool = False,
                 verbose: bool = False) -> Tuple[Sum, pd.DataFrame]:
        if sql:
             answer, code_labels = self._execute_sql(simplified=simplified)
             self.mapper = code_labels
        else:
            measure_units = {}
            if isinstance(self.sub_expression, Value) and len(self.selectors) == 0:
                measure_units = engine.validate_msr_unit_compatibility(self.sub_expression.measures)
            sub_exp, table = self.sub_expression(offline=offline, odata4=odata4, verbose=verbose)

            self.mapper = sub_exp.mapper
            self.intermediate_results = [table] + sub_exp.intermediate_results

            measure_units = {self.mapper.get(k, k): v for k, v in measure_units.items()}
            answer = self._aggregate(table, self.selectors, pd.DataFrame.sum, measure_units)

        if verbose:
            self._print_answer(answer)

        return self, answer

    @property
    def odata3_sql_simplified(self) -> str:
        """
            === Example simplified SUM expression as OData3 SQL ===
            SELECT 'Livestock_1' AS Measure, Periods,
                   SUM(Livestock_1) AS 'SUM'
            FROM '<parquet-files>'
            WHERE FarmAnimals IN ('A044005', 'A044022')
            AND Periods IN ('2022MM04')
            GROUP BY Periods;
        """
        if not isinstance(self.sub_expression, Value):  # Aggregate over complex inner query (e.g. AGGJOIN)
            if len(self.selectors) == 0:
                _, _, measures, _ = self._get_sub_exp_filters()
                sql = f"""
                    SELECT
                        '{str(list(measures)[0])}' AS Measure,
                        SUM({str(list(measures)[0])}) AS SUM
                    FROM ({self.sub_expression.odata3_sql_simplified})
                    GROUP BY Measure
                """
                return sqlglot.parse_one(sql).sql(pretty=True)
            else:
                raise NotImplementedError("Can't build SQL aggregations over complex inner queries "
                                          "with a dimension as the aggregation selector yet.")

        value_exps, tables, measures, dimensions = self._get_sub_exp_filters()
        where_stmts = []
        for sub_exp in value_exps:
            where_stmts.append(sqlglot.parse_one(sub_exp.odata3_sql_simplified).find(sqlglot.exp.Where).this)

        sql = f"""
            SELECT
                {', '.join(str(group) for group, _ in dimensions) + ',' if len(dimensions) > 0 else ''}
                {self._operator}("{'" + "'.join(map(str, measures))}") AS '{self._operator}'
            FROM '{os.path.relpath(config.DB_ODATA3_FILES)}/{list(tables)[0]}.parquet'
            {sqlglot.exp.Where(this=sqlglot.expressions.and_(*where_stmts)).sql()}
            GROUP BY {', '.join(str(group) for group, _ in dimensions)}
        """
        return sqlglot.parse_one(sql).sql(pretty=True)


if __name__ == '__main__':
    from s_expression.parser import parse, eval

    # Measure sum
    s_expr, answer = eval(parse("""(SUM
        ()
        (VALUE 83180NED
            (MSR (D004585_1 D004560_1 D004550_1))
            (DIM Perioden (2015JJ00 2016JJ00 2017JJ00 2018JJ00 2019JJ00 2020JJ00 2021JJ00 2022JJ00))
        )
    )
    """), offline=False, verbose=True)

    # Dim sum --- not to be confused with much more fun Chinese dim sum
    eval(parse("""(SUM
         (Perioden)
         (VALUE 85302NED 
             (MSR (D004645 D006211_2))
             (DIM Perioden (2021JJ00 2022JJ00))
             (DIM BestemmingEnSeizoen (L008691 L999996))
             (DIM Vakantiekenmerken (T001460))
             (DIM Marges (MW00000))
         )
     """), sql=False, verbose=True)

    # Sum with join
    eval(parse("""
        (SUM ()
            (JOIN (M004032) (VALUE 85601NED (MSR (M004032)) (DIM ContainerGrootte (A052183 A052184)) (DIM Vervoerstroom (A045748)) (DIM Perioden (2023JJ00))) (VALUE 85598NED (MSR (M004032)) (DIM NederlandseZeehavens (T001293)) (DIM SoortLading (A041789)) (DIM Vervoerstroom (A045748)) (DIM Perioden (2023JJ00))))
        )
    """), sql=True, verbose=True)

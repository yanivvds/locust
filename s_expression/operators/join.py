from __future__ import annotations

import inspect
import pandas as pd
import sqlglot
from beartype import beartype
from copy import deepcopy
from itertools import product
from typing import List, Tuple, Set, FrozenSet

from odata_graph import engine
from pipeline.db_executor import DBExecutor
from pipeline.odata_executor import CodeLabelMapper
from s_expression import Expression, ParsedExpression, Table, Measure, Dimension
from s_expression.operators import Value, Sum, Avg, Min, Max
from utils.custom_types import NonEmptyList


class Join(Expression):
    """
        The JOIN expression does an outer join of two or multiple intermediate pivot tables,
        resulting from a set of sub-expressions, over given axes. The axes will be used as the
        index of the pivot tables from the evaluated sub-expressions.

        :param sexp: the parsed S-expression, of which the evaluation instantiates this object
        :param selector: axes to join the intermediate pivot table results on
        :param sub_expressions: VALUE expressions that will be executed and used for joining
    """
    @beartype
    def __init__(self, sexp: ParsedExpression, selector: NonEmptyList[str], *sub_expressions: Value):
        super().__init__(sexp)
        self.sub_expressions = sub_expressions

        # Find out if the selector is a measure or a dim group
        self.measure_selector = any(s in map(str, set().union(*[sexp.measures for sexp in self.sub_expressions])) for s in selector)
        self.select_cols = selector
        self.selectors = selector if not self.measure_selector else {'Measure', 'Unit'}

    def __call__(self,
                 sql: bool = False,
                 odata4: bool = False,
                 simplified: bool = False,
                 offline: bool = False,
                 verbose: bool = False) -> Tuple[Join, pd.DataFrame]:
        if sql:
            tables = []
            measures = set()
            dims = set()
            for s in self.sub_expressions:
                # Create a deepcopy, for the DBExecutor will otherwise change the original sub_exp properties as well
                sub_exp = deepcopy(s)
                value_exp = sub_exp
                while 1:  # Get inner VALUE expression
                    if isinstance(value_exp, Value):
                        break
                    value_exp = sub_exp.sub_expression

                tables.append(sub_exp.table)
                measures |= sub_exp.measures
                dims |= sub_exp.dimensions

            db = DBExecutor(tables=tables, measures=measures, dims=dims, operator_name=self._operator)
            if simplified:
                query = self.odata3_sql_simplified
            else:
                query = self.odata3_sql

            answer, code_labels, _ = db.query_db(query=query, index_cols=frozenset(self.selectors), simplified=simplified)
            self.mapper = code_labels
        else:
            self.intermediate_results = []
            self.mapper = CodeLabelMapper({})
            measures = set()
            for sexp in self.sub_expressions:
                sub_exp, table = sexp(index_cols=frozenset(self.selectors), friendly_labels=False,
                                      offline=offline, odata4=odata4, verbose=verbose)
                measures |= sub_exp.measures
                self.intermediate_results.append(table)
                self.mapper |= sub_exp.mapper

            answer = pd.concat(self.intermediate_results, axis=1)

            # Translate the (merged) code columns and indices to their friendly names
            answer.index = [tuple(self.mapper.get(i.strip(), i) for i in p_i) if not isinstance(p_i, str)
                            else self.mapper.get(p_i.strip(), p_i) for p_i in answer.index]
            answer.index.names = list({'Measure'} if self.measure_selector else self.selectors)
            answer.columns = [tuple(self.mapper.get(c.strip(), c) for c in p_col) if not isinstance(p_col, str)
                              else self.mapper.get(p_col.strip(), p_col) for p_col in answer.columns]

        # If JOIN was called from within an aggregator, check for measure compatibility
        called_from_aggregator = any(
            # Checking for SimpleAggregator here results in a circular import :(
            isinstance(f_info.frame.f_locals.get('self'), (Sum, Avg, Min, Max)) for f_info in inspect.stack())
        if called_from_aggregator and self.measure_selector:
            # TODO: report measure_units back for the aggregators to use when they need to multiply stuff
            _ = engine.validate_msr_unit_compatibility(measures)

        if verbose:
            self._print_answer(answer)

        return self, answer

    def _get_sub_exp_filters(self) -> Tuple[List[Value], Set[Table], Set[Measure], Set[Tuple[Dimension, FrozenSet[Dimension]]]]:
        """
            Helper function for getting all measures and dimensions from all
            inner VALUE expressions in this S-expression, regardless of their depth.

            :returns: Tuple containing a list of inner VALUE expressions, set with tables,
                      set with measures and set with dimensions
        """
        tables = set()
        measures = set()
        dimensions = set()
        value_exps = []
        for sub_exp in self.sub_expressions:
            inner_expressions: List[Expression] = [sub_exp]
            for inner_exp in inner_expressions:
                if isinstance(inner_exp, Value):
                    tables |= {inner_exp.table}
                    measures |= inner_exp.measures
                    dimensions |= inner_exp.dimensions
                    value_exps.append(inner_exp)
                    continue

                if hasattr(inner_exp, 'sub_expressions'):
                    inner_expressions.extend(inner_exp.sub_expressions)
                else:
                    inner_expressions.append(inner_exp.sub_expression)

        return value_exps, tables, measures, dimensions

    @property
    def odata3_sql(self):
        """
            === Example JOIN s-expression as SQL ===
            SELECT CONCAT_WS(', ', TableA.<selectorA>, TableA.<selectorB>) AS Dimension_Measure,
                   <measure-col1> <measure-col2>
            FROM (
                <sub-query>
            ) AS TableA
            INNER JOIN (
                <sub-query>
            ) AS TableB
            ON TableA.selector = TableB.selector;
        """
        inner_table_sqls = [sqlglot.parse_one(sub.odata3_sql) for sub in self.sub_expressions]

        # Get measures to select per table
        measures = []
        for inner_sql in inner_table_sqls:
            table_measures = []
            for col in inner_sql.find_all(sqlglot.exp.In):
                if 'Measure' in col.sql():
                    for msr in [s.name for s in col.expressions]:
                        table_measures.append(msr)
            measures.append(table_measures)

        # Remove selector from PIVOTs and add as WHERE clause and determine if the selector is a measure
        for_statements = []
        for i, table in enumerate(inner_table_sqls):
            pivot = table.find(sqlglot.exp.Pivot)
            table_fors = []
            for node in pivot.args['fields']:
                if isinstance(node, sqlglot.exp.In):
                    if not any(s in node.sql() for s in self.selectors):  # Non-selector, keep in PIVOT
                        table_fors.append(node)
                    else:  # JOIN selector, move to where of inner SQL for filtering
                        inner_select = table.find(sqlglot.exp.Subquery).find(sqlglot.exp.Select)
                        inner_select.where(node.sql(), append=True, copy=False)

            if len(table_fors) == 0:
                # Default to pivot on Measure if all table dimensions are used as join selectors
                table_fors.append(sqlglot.parse_one(f"Measure IN ('{"', '".join(measures[i])}')"))

            if len(table_fors) > 0:
                for_statements.append(table_fors)
            pivot.set('fields', table_fors)

        # Fill the SELECT and JOIN-ON statements with the proper columns
        if not self.measure_selector:
            if len(self.selectors) > 1:
                select = "CONCAT_WS(', ', TableA."
                select += ', TableA.'.join(self.selectors) + ') AS Dimension_Measure, '
                select += ', '.join([msr for table_measures in measures for msr in table_measures])
            else:
                select = f"TableA.{self.selectors[0]}"
                for i, fors in enumerate(for_statements):
                    combs = product(*[[s.name for s in f.expressions] for f in fors])
                    select += f", {', '.join(['_'.join(c) for c in combs])}"

            on_selectors = []
            for selector in self.selectors:
                on_selectors.append(f"TableA.{selector} = TableB.{selector}")
            join_on = ' AND '.join(on_selectors)
        else:
            select = "*"
            join_on = "TableA.Measure = TableB.Measure"

        sql = f"""
            SELECT {select}
            FROM (
                {inner_table_sqls[0]}
            ) AS TableA
            FULL JOIN (
                {inner_table_sqls[1]}
            ) AS TableB
            ON {join_on}
        """
        return sqlglot.parse_one(sql).sql(pretty=True)

    @property
    def odata3_sql_simplified(self):
        """
            === Example simplified JOIN expressions as SQL (OData3) ===
            SELECT COALESCE(TableA.Periods, TableB.Periods) AS 'Periods',
                   BankruptciesSessionDayCorrected_1,
                   SoldHomes_4
            FROM (
                SELECT *
                FROM '<parquet-file>'
                WHERE Periods IN ('2007KW04')
                AND TypeOfBankruptcy IN ('A041718', 'A047597')
            ) AS TableA
            FULL JOIN (
                SELECT *
                FROM '<parquet-file>'
                WHERE Periods IN ('2007KW04')
            ) AS TableB
            ON TableA.Periods = TableB.Periods;
        """
        inner_table_sqls = [sqlglot.parse_one(sub.odata3_sql_simplified) for sub in self.sub_expressions]
        for inner_sql in inner_table_sqls:
            inner_sql.find(sqlglot.exp.Select).args['expressions'] = [sqlglot.exp.Star()]

        measures_per_inner = [sub_exp.measures for sub_exp in self.sub_expressions]
        dimensions_per_inner = [sub_exp.dimensions for sub_exp in self.sub_expressions]

        # Fill the SELECT and JOIN-ON statements with the proper columns
        select_cols = ""
        for col in self.select_cols:
            if col.replace('_', '').isnumeric():
                select_cols += f"COALESCE(TableA.\"{col}\", TableB.\"{col}\") AS '{col}',\n"
            else:
                select_cols += f"COALESCE(TableA.{col}, TableB.{col}) AS '{col}',\n"
        select_cols = select_cols[:-2]  # remove trailing comma

        for table, measures in zip(['TableA', 'TableB'], measures_per_inner):
            for measure in measures:
                if str(measure) not in select_cols:
                    if str(measure).replace('_', '').isnumeric():
                        select_cols += f", {table}.\"{str(measure)}\""
                    else:
                        select_cols += f", {table}.{str(measure)}"

        for table, dimensions in zip(['TableA', 'TableB'], dimensions_per_inner):
            for dim, _ in dimensions:
                if str(dim) not in select_cols:
                    if str(dim).replace('_', '').isnumeric():
                        select_cols += f", {table}.\"{dim}\""
                    else:
                        select_cols += f", {table}.{dim}"

        on_selectors = []
        common_dims = (set(map(str, [g for g, _ in dimensions_per_inner[0]])) &
                       set(map(str, [g for g, _ in dimensions_per_inner[1]])))
        for dim in common_dims:
            on_selectors.append(f"TableA.{dim} = TableB.{dim}")

        if self.measure_selector:
            common_measures = set(map(str, measures_per_inner[0])) & set(map(str, measures_per_inner[1]))
            for measure in common_measures:
                if str(measure).replace('_', '').isnumeric():
                    on_selectors.append(f"TableA.\"{str(measure)}\" = TableB.\"{str(measure)}\"")
                else:
                    on_selectors.append(f"TableA.{str(measure)} = TableB.{str(measure)}")

        if len(on_selectors) == 0:
            for col in self.select_cols:
                if col.replace('_', '').isnumeric():
                    on_selectors.append(f"TableA.\"{col}\" = TableB.\"{col}\"")
                else:
                    on_selectors.append(f"TableA.{col} = TableB.{col}")
        join_on = ' AND '.join(on_selectors)

        sql = f"""
            SELECT {select_cols}
            FROM (
                {inner_table_sqls[0]}
            ) AS TableA
            FULL JOIN (
                {inner_table_sqls[1]}
            ) AS TableB
            ON {join_on}
        """
        return sqlglot.parse_one(sql).sql(pretty=True)


if __name__ == '__main__':
    from s_expression.parser import parse, eval

    eval(parse("""
        (JOIN (Regions) (VALUE 80783eng (MSR (CattleTotal_103)) (DIM FarmTypes (A009518 A009481 A009510)) (DIM Regions (LG12  )) (DIM Periods (2006JJ00))) (VALUE 80783eng (MSR (CattleTotal_103)) (DIM FarmTypes (A009518 A009481 A009510)) (DIM Regions (LG12  )) (DIM Periods (2007JJ00))) (VALUE 80784eng (MSR (RegularlyEmployedTotal_19)) (DIM Regions (LG12  )) (DIM Gender (3000  )) (DIM Periods (2014JJ00))))
    """), verbose=True)

    eval(parse("""
        (JOIN (M004032) (VALUE 85601NED (MSR (M004032)) (DIM ContainerGrootte (A052183 A052184)) (DIM Vervoerstroom (A045748)) (DIM Perioden (2023JJ00))) (VALUE 85598NED (MSR (M004032)) (DIM NederlandseZeehavens (T001293)) (DIM SoortLading (A041789)) (DIM Vervoerstroom (A045748)) (DIM Perioden (2023JJ00))))
    """), verbose=True)

    eval(parse("""
        (JOIN (Perioden) (VALUE 85601NED (MSR (M004032)) (DIM ContainerGrootte (A052183 A052184)) (DIM Vervoerstroom (A045748)) (DIM Perioden (2023JJ00))) (VALUE 85598NED (MSR (M004032)) (DIM NederlandseZeehavens (T001293)) (DIM SoortLading (A041789)) (DIM Vervoerstroom (A045748)) (DIM Perioden (2023JJ00))))
    """), verbose=True)

    eval(parse("""
        (JOIN (Regions) (VALUE 80783eng (MSR (CattleTotal_103)) (DIM FarmTypes (A009518 A009481 A009510)) (DIM Regions (LG12  )) (DIM Periods (2006JJ00))) (VALUE 80784eng (MSR (RegularlyEmployedTotal_19)) (DIM Regions (LG12  )) (DIM Gender (3000  )) (DIM Periods (2014JJ00))))
    """), verbose=True)

    eval(parse("""
        (JOIN (Perioden)
            (VALUE 84957NED
                (MSR (M004367))
                (DIM Perioden (2021JJ00 2022JJ00))
                (DIM Vervoerstromen (A045747))
            )
            (VALUE 85302NED
                (MSR (D006211_2))
                (DIM Perioden (2021JJ00 2022JJ00))
                (DIM BestemmingEnSeizoen (T001047))
                (DIM Vakantiekenmerken (T001460))
                (DIM Marges (MW00000))
            )
        )
    """), verbose=True)

from __future__ import annotations

import os
import pandas as pd
import sqlglot
from beartype import beartype
from beartype.door import is_bearable
from typing import List, Set, FrozenSet, Tuple, Union, Optional

import config
from pipeline.odata_executor import ODataExecutor
from pipeline.db_executor import DBExecutor
from s_expression import Expression, ParsedExpression
from s_expression import Table, Measure, Dimension
from utils.custom_types import NonEmptyList, ComparisonOperator


class Value(Expression):
    """
        Basic value expression used for retrieving and filtering information from OData
        Syntax: (VALUE <table> (MSR (<code> ... <code>)) (DIM <dim> (<code> ... <code>)) (DIM ...))

        :param sexp: the parsed S-expression, of which the evaluation instantiates this object
        :param table: the table identifier to query
        :param measures: the measures identifier to filter the table on
        :param dimensions: the dimensions and their codes to filter the table on
    """
    table: Table
    measures: Set[Measure]
    measure_filter: Optional[Tuple[Measure, ComparisonOperator, float]]
    dimensions: Set[Tuple[Dimension, FrozenSet[Dimension]]]

    @beartype
    def __init__(self,
                 sexp: ParsedExpression,
                 table: str,
                 measures: NonEmptyList[Union[str, List[str]]],
                 *dimensions: Tuple[str, List[str]]):
        super().__init__(sexp)
        self.table = Table(table)

        self.measures = set()
        self.measure_filter = None
        i = 0
        while i < len(measures):
            if i + 2 < len(measures) and is_bearable(measures[i + 1], ComparisonOperator):
                try:
                    code, op, value = measures[i], measures[i + 1], float(measures[i + 2])
                except ValueError:
                    raise ValueError(f"Invalid measure specification: comparison value '{measures[i + 2]}' must "
                                     f"be numeric in `{measures[i]} {measures[i + 1]} {measures[i + 2]}`")
                self.measures.add(Measure(code))
                self.measure_filter = (Measure(code), op, value)

                if len(measures) > 3:
                    raise TypeError(f"Invalid measure specification {measures}.\n"
                                    f"You can only select one measure when filtering by comparison operator.")
                break
            else:
                code = measures[i]
                self.measures.add(Measure(code))
                i += 1

        self.selectors = {}  # only relevant when VALUE is called from within a different operator
        self.intermediate_results = []

        # Ideally we would specify the type signature for dimensions as List[Tuple[str, Tuple[str, ...]]],
        #  following PEP646, but this isn't (yet) supported by beartype .
        #  (https://github.com/beartype/beartype/issues/428#issuecomment-2357650766)
        if not (all(map(lambda x: len(x) == 2, dimensions))  # every dimension must contain two items
                and all(map(lambda x: isinstance(x[0], str), dimensions))  # every first item must be a single string
                and all(map(lambda x: is_bearable(x[1], List[str]), dimensions))  # every second item must be a list of strings
        ):
            raise TypeError(f"Invalid dimension specification {dimensions}.\n"
                            f"Please make sure specify dimensions as `(DIM <dim> (<code> ... <code>))`.")

        self.dimensions = {(Dimension(d), frozenset({Dimension(c) for c in codes})) for d, codes in dimensions}

    def __call__(self,
                 index_cols: FrozenSet[str] = frozenset({'Measure', 'Unit'}),
                 friendly_labels: bool = True,
                 sql: bool = False,
                 odata4: bool = False,
                 simplified: bool = False,
                 offline: bool = False,
                 verbose: bool = False) -> Tuple[Value, pd.DataFrame]:
        """
            :param index_cols: column to index over when creating the answer pivot table
            :param friendly_labels: flag whether to translate measure/dimension codes in the answer to their labels
        """
        if sql or offline:
            self.selectors = index_cols or frozenset({'Measure', 'Unit'})
            db = DBExecutor(tables=[self.table], measures=self.measures, dims=self.dimensions,
                            operator_name=self._operator)

            if simplified:
                query = self.odata3_sql_simplified
            else:
                query = self.odata3_sql

            answer, code_labels, _ = db.query_db(query=query, index_cols=index_cols,
                                                 simplified=simplified, friendly_labels=friendly_labels)
        else:
            odata = ODataExecutor(self.table, self.measures, self.measure_filter, self.dimensions)
            answer, code_labels = odata.query_odata(index_cols=index_cols, friendly_labels=friendly_labels)

        self.mapper = code_labels
        if verbose:
            self._print_answer(answer)

        return self, answer

    @property
    def odata3_sql(self):
        """
            === Example VALUE s-expression as OData3 SQL ===
            SELECT *
            FROM (
                SELECT Measure, Value, ArableCrops, Regions, Periods
                FROM '<parquet_file>'
                UNPIVOT (
                    Value FOR Measure IN (HarvestedArea_2, GrossYieldPerHa_3)
                )
            )
            PIVOT(MAX(Value) FOR
                ArableCrops IN ('A042362', 'A042180')
                Periods IN ('2013JJ00')
                Regions IN ('PV29')
            );
        """
        measure_selector = any(s in map(str, self.measures) for s in self.selectors)
        if measure_selector:
            where_cols = [('Measure', self.selectors)]
        else:
            where_cols = [(group, codes) for group, codes in self.dimensions if str(group) in self.selectors]

        where = 'WHERE '
        if where_cols:
            where += ' AND '.join(
                f"{str(group)} IN ('{'\', \''.join([str(c) for c in codes])}')" for group, codes in where_cols
            )

        # Comparison filters
        if self.measure_filter:
            if where!= 'WHERE ':
                where += ' AND '
            _, op, val = self.measure_filter
            where += f"Value {op} {val}"

        sql = f"""
            SELECT *
            FROM (
                SELECT Measure, Value, {', '.join(str(g) for g, _ in self.dimensions)}
                FROM '{os.path.relpath(config.DB_ODATA3_FILES)}/{self.table}.parquet'
                UNPIVOT (
                    Value FOR Measure IN ('{"', '".join(str(msr) for msr in self.measures)}')
                )
                {where if where != "WHERE " else ""}
            )
            PIVOT (
                MAX(Value)
                FOR {'\n'.join('{} IN {}'.format(group, f"('{"', '".join({str(c) for c in codes})}')")
                               for group, codes in self.dimensions | ({('Measure', frozenset(self.measures))} if not measure_selector else set())
                               if str(group) not in self.selectors and len(codes) > 0)}
            )
        """
        return sqlglot.parse_one(sql).sql(pretty=True)

    @property
    def odata3_sql_simplified(self):
        """
            === Example simplified VALUE expression as OData3 SQL ===
            SELECT Period, TypeOfBankruptcy, BankruptciesSessionDayCorrected_1
            FROM '<parquet_file>'
            WHERE TypeOfBankruptcy = 'A047597'
            AND Periods = '1993MM09';
        """
        select_cols = ""
        if self.dimensions:
            select_cols += ", ".join(map(lambda d: str(d[0]), self.dimensions)) + ', '
        select_cols += ', '.join(map(str, self.measures))

        sql = f"""
            SELECT {select_cols}
            FROM '{os.path.relpath(config.DB_ODATA3_FILES)}/{self.table}.parquet'
        """
        where_statements = []
        for group, codes in self.dimensions:
            if len(codes) > 1:
                where_statements.append(f"{group} IN ('{"', '".join({str(c) for c in codes})}')")
            if len(codes) == 1:
                where_statements.append(f"{group} = '{list(codes)[0]}'")

        where = ""
        if len(where_statements) > 0:
            where += 'WHERE '
            where += '\nAND '.join(where_statements)

        # Comparison filters
        if self.measure_filter:
            if where != 'WHERE ':
                where += ' AND '
            msr, op, val = self.measure_filter
            where += f"\"{msr}\" {op} {val}"

        sql += where
        return sqlglot.parse_one(sql).sql(pretty=True)


if __name__ == '__main__':
    from s_expression.parser import parse, eval

    s_expression, answer = eval(parse("""(VALUE 85302NED
        (MSR (D004645 < 35000))
        (DIM BestemmingEnSeizoen (T001047))
        (DIM Vakantiekenmerken (T001460))
        (DIM Marges (MW00000))
    )
    """), verbose=True)

    eval(parse("""(VALUE 85302NED
        (MSR (D004645 < 35000))
        (DIM Perioden (2021JJ00 2022JJ00 2023JJ00 2024JJ00))
        (DIM BestemmingEnSeizoen (T001047))
        (DIM Vakantiekenmerken (T001460))
        (DIM Marges (MW00000))
    )
    """), verbose=True)

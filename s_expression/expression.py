from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod
from re import sub
from typing import Tuple, List, Dict, Union, TypeVar, Callable

import config
from s_expression.mapper import CodeLabelMapper

T = TypeVar("T")
ParsedExpression = List[Union[str, T]]


class Expression(ABC):
    """Main abstract node type of syntax tree for s-expressions."""
    SOS = '('  # Start of S-expression
    EOS = ')'  # End of S-expression

    mapper: CodeLabelMapper
    sub_expression: Expression

    @abstractmethod
    def __init__(self, parsed_sexp: ParsedExpression):
        self._parsed_sexp = parsed_sexp
        self.intermediate_results = []

    @abstractmethod
    def __call__(self,
                 sql: bool = False,
                 odata4: bool = False,
                 simplified: bool = False,
                 offline: bool = False,
                 verbose: bool = False) -> Tuple[Expression, pd.DataFrame]:
        """
            The logic when executing the sub-expression. Every expression should return a DataFrame,
            i.e. the pivot table that is the actual answer, and a mapping dict where the friendly
            names in the answer table can be translated back to their raw codes for further processing.

            :param sql: flag whether to execute the query as SQL (True) or as native S-expression (False; default)
            :param odata4: flag indicating whether to search SQL using OData4 (True) of OData3 (False; default)
            :param simplified: flag indicating whether to use simplified OData3 SQLs (True) or not (False; default)
            :param offline: flag indicating whether to use the OData API service (False; default) or offline evaluation
                            of VALUE expressions using SQL (True). Only relevant when sql=False.
            :param verbose: show the (intermediate) outputs of the executed expression/query
            :returns: (self,
                       DataFrame containing the retrieved and processed answer,
                       intermediate answers,
                       measure/dimension code to label mapping)
        """
        raise NotImplementedError()

    def _aggregate(self, table: pd.DataFrame, groupby: List[str], agg_func: Callable, measure_units: Dict[str, Dict[str, str]] = None) -> pd.DataFrame:
        """
            Do the aggregation and execute a given function to the given pivot table.

            :param table: The pivot table to aggregate. This should be the result of a VALUE function.
            :param groupby: The columns to aggregate over.
            :param agg_func: The Pandas aggregation function to use. (e.g. pd.DataFrame.sum, pd.DataFrame.mean, etc.)
            :param measure_units: dictionary containing unit information for all measures in the table
            :returns: new resulting pivot table as a DataFrame.
        """
        # If aggregating on measures, apply conversion multipliers when not equal
        if len(groupby) == 0:
            measure_units = measure_units or {}
            multipliers = [d['multiplier'] for d in measure_units.values()]
            if len(set(multipliers)) > 1:
                for msr, units in measure_units.items():
                    mask = table.index.get_level_values('Measure') == msr
                    table.loc[mask] *= int(units['multiplier'] or 1)
                    new_multiplier = 'aantal' if config.LANGUAGE == 'nl' else 'number'
                    table.index = table.index.set_levels([new_multiplier] * len(multipliers), level='Unit', verify_integrity=False)

        if len(groupby) == 0:
            # Aggregate over all measures when no selectors are given
            agg = pd.DataFrame(agg_func(table, axis=0 if len(table.index) > 1 else 1)).T
            agg.index = ['\n'.join(table.index.get_level_values('Measure').map(str))]
            return agg

        # Throw exception if the groupby column(s) are not present in the pivot table
        for g in groupby:
            if self.mapper[g] not in table.columns.names:
                loc = [' '] * len(str(self))
                loc[str(self).index(g):str(self).index(g) + len(g)] = ['^'] * len(g)
                raise ValueError(f"Can't aggregate over column {g} in the expression"
                                 f"\n{self}\n{''.join(loc)}\n"  # totally useless but nice visual error indication
                                 f"Make sure {g} is present in the requested table in the VALUE expression.")

        group_cols = [c for c in table.columns.names if self.mapper.inv.get(c) not in groupby]  # inverted group by
        if len(group_cols) == 0:
            # Aggregation logic for single dimension tables
            agg = agg_func(table, axis=1)
            agg = pd.DataFrame(agg, columns=[f"{self._operator}{[self.mapper[groupby[0]]]}"])
        else:
            # Do an inverted group by over the selected cols, such that the column names of
            #  the columns not in the group by can be re-added for clarity
            agg = table.T.groupby(group_cols).apply(agg_func)
            agg.index = [(tuple([i]) if isinstance(i, str) else i) + (f"{self._operator}{[self.mapper[s] for s in groupby]}",)
                         for i in agg.index]

            agg = agg.T

        return agg

    @property
    def _operator(self) -> str:
        """The function name in the written S-expression."""
        return self.__class__.__name__.upper()

    @staticmethod
    def format_answer(answer: pd.DataFrame):
        markdown = answer.to_markdown(headers=['\n'.join([str(c) for c in c])
                                               if isinstance(c, Tuple) else c for c in answer.columns.values],
                                      tablefmt='simple_grid')
        return markdown

    def _print_answer(self, answer: pd.DataFrame):
        """Pretty print a DataFrame pivot table."""
        print(str(self))
        print(self.format_answer(answer))

    @property
    def _sexp(self) -> str:
        """Parsable and executable Lisp/string version of this expression."""
        trans = str.maketrans('[]', '()', "',")
        return str(self._parsed_sexp).translate(trans)

    @property
    def _friendly_sexp(self) -> str:
        """
            Returns the S-expression with the codes substituted for their friendly labels.
            Requires the expression to be evaluated first in order to fetch the code-label mapping.
        """
        if self.mapper is None:
            raise ValueError("No mapper defined yet. Make sure to evaluate the expression first.")

        s = self._sexp
        for code, label in self.mapper.items():
            s = s.replace(code, f"'{label}'")
        return s

    @property
    @abstractmethod
    def odata3_sql(self) -> str:
        """Parsable and executable OData3 SQL version of this expression."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def odata3_sql_simplified(self) -> str:
        """Simplified (non-pivoted) version of the OData3 SQL version of this expression."""
        raise NotImplementedError()

    def __str__(self) -> str:
        return self._sexp

    def __repr__(self) -> str:
        tupelize = lambda l: tuple(tupelize(x) for x in l) if isinstance(l, list) else l
        return sub(r"[',]", '', str(tupelize(self._parsed_sexp)))

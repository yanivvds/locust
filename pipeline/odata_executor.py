"""
    This module retrieves the data from an S-expression and returns a pivot table as a pandas DataFrame.
    This module is similar to the ` db_executor`, which does the same for SQL queries.
"""
import numpy as np
import operator
import pandas as pd
from rdflib import SKOS, QB, DCTERMS as DCT
from typing import Set, FrozenSet, Tuple, Optional

import config
from odata_graph import engine
from odata_graph.sparql_controller import QUDT
from s_expression.mapper import CodeLabelMapper
from s_expression import Table, Measure, Dimension, uri_to_code
from utils.custom_types import ComparisonOperator, UnitCompatibilityError
from utils.global_functions import secure_request

COMPARISON_OPERATORS = {
    '<': operator.lt,
    '>': operator.gt,
    '!=': operator.ne,
    '<=': operator.le,
    '>=': operator.ge,
    '=': operator.eq
}

class OData3QueryBuilder(object):
    """
        Helper class for generating a OData3 request/query to obtain
        observations of a table using given measure and dimension filters.
    """
    filters = False

    def __init__(self, table_id: Table):
        self.query = f"{config.ODATA3_BASE_URL}/{table_id}/TypedDataSet"

    def add_selects(self, measures: Set[Measure], dim_groups: Set[Dimension]):
        self.query += f'''&$select=ID, {", ".join([str(s) for s in measures | dim_groups])}'''

    def add_dim_filter(self, group: Dimension, codes: Set[Dimension]):
        if len(codes) > 0:
            self.query += '?$filter=' if not self.filters else ' and '
            self.query += f'''((substringof({')) or (substringof('.join(["'" + str(dim) + "'," + str(group) for dim in codes])})))'''
            self.filters = True

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.query


class OData4QueryBuilder(object):
    """
        Helper class for generating a OData4 request/query to obtain
        observations of a table using given measure and dimension filters.
    """
    filters = False

    def __init__(self, table_id: Table):
        self.query = f"{config.ODATA4_BASE_URL}/{table_id}/Observations/"

    def add_msr_filter(self, measures: Set[Measure]):
        self.query += '?$filter=' if not self.filters else ' and '
        self.query += f'''Measure in ('{"', '".join([str(m) for m in measures])}')'''
        self.filters = True

    def add_dim_filter(self, group: Dimension, codes: Set[Dimension]):
        if len(codes) > 0:
            self.query += '?$filter=' if not self.filters else ' and '
            # There's no distinction between codes of given dim group and all codes, but that shouldn't matter
            self.query += f'''{group} in ('{"', '".join([str(dim) for dim in codes])}')'''
            self.filters = True

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.query


class ODataExecutor(object):
    """
        Object for handling OData4 requests, following the execution of a VALUE expression.
        Requires a table identifier and a set of measure and dimension identifiers for filtering.

        :param table: URI and identifier of table to fetch
        :param measures: set of measure URIs to filter table on
        :param dims: set of dimension URIs to filter table on
    """
    def __init__(self,
                 table: Table,
                 measures: Set[Measure],
                 measure_filter: Optional[Tuple[Measure, ComparisonOperator, float]],
                 dims: Set[Tuple[Dimension, FrozenSet[Dimension]]]):
        self.table = table
        self.measures = measures
        self.measure_filter = measure_filter
        self.dims = dims

        if config.LANGUAGE == 'nl':
            self._odata4 = True
        else:
            self._odata4 = False

        self._odata_query = None
        self._graph = engine.get_table_graph(table=self.table, include_time_geo_dims=True)

    def query_odata(self,
                    index_cols: FrozenSet = frozenset({'Measure', 'Unit'}),
                    friendly_labels: bool = True) -> Tuple[pd.DataFrame, CodeLabelMapper]:
        """
            Retrieve values from OData4 and create a pivot table using table, measure and dimension selectors.

            :param index_cols: the columns to use as the MultiIndex for the output pivot tables.
                               Defaults to ['Measure', 'Unit']
            :param friendly_labels: map the OData4 codes in the output table to their friendly names.
                                    Set to False when wanting to JOIN the pivot tables at a later stage.
            :returns: tuple containing a DataFrame pivot table and mapper for codes to labels
        """
        table = self.table

        # Get friendly names of codes
        concept_labels = dict(self._graph.subject_objects(SKOS.prefLabel))
        code_labels = CodeLabelMapper(
            # MSR/DIM-Concept mapping
            {str(obs).split(Measure.rdf_ns if obs in Measure.rdf_ns else Dimension.rdf_ns)[-1]: str(concept_labels[concept])
             for obs, concept in self._graph.subject_objects(QB.concept)} |
            # Table title
            {str(table).split(Table.rdf_ns)[-1]: str(title)
             for table, title in self._graph.subject_objects(DCT.title)}
        )

        if self._odata4:  # Dutch
            self._odata_query = OData4QueryBuilder(table)
            self._odata_query.add_msr_filter(self.measures)
            for dim in self.dims:
                self._odata_query.add_dim_filter(dim[0], set(dim[1]))
        else:  # English, OData3
            self._odata_query = OData3QueryBuilder(table)
            for dim in self.dims:
                self._odata_query.add_dim_filter(dim[0], set(dim[1]))
            self._odata_query.add_selects(self.measures, {d[0] for d in self.dims})

        try:
            observations = secure_request(str(self._odata_query), json=True, max_retries=9, timeout=8)

            if not observations or not observations['value']:
                raise ValueError("Retrieved OData data is empty.")
            else:
                observations = observations['value']

            if not self._odata4:
                # Translate OData3 result to OData4
                msr_str = {str(m) for m in self.measures}
                odata4_observations = []

                for obs in observations:
                    for i, msr in enumerate(msr_str):
                        odata4_obs = {k: v for k, v in obs.items() if k not in msr_str}
                        odata4_obs['ID'] = f"{obs['ID']}_{i}"
                        odata4_obs['Measure'] = msr
                        odata4_obs['Value'] = float(obs[msr] if obs[msr] is not None else np.nan)
                        odata4_observations.append(odata4_obs)

                observations = odata4_observations

            # Filter values
            # CBS' API denies the Value filter in OData4 and Measure filters in OData3, so filtering needs
            #  to happen client-side
            if self.measure_filter:
                msr, op, value = self.measure_filter
                observations = [obs for obs in observations if obs['Measure'] == str(msr)
                                and COMPARISON_OPERATORS[op](obs['Value'], value)]

            result_df = pd.DataFrame()
            if len(observations) > 0:
                obs_df = pd.DataFrame(observations)
                obs_df = obs_df.set_index('Id' if self._odata4 else 'ID', drop=True)
                if self._odata4:
                    obs_df = obs_df.drop(['ValueAttribute', 'StringValue'], axis=1)

                units = {uri_to_code(m): str(u) for m, u in self._graph.subject_objects(QUDT.unitOfSystem)}
                obs_df['Unit'] = obs_df['Measure'].map(units).replace({np.nan: ''})

                if friendly_labels:
                    obs_df = obs_df.rename(columns=code_labels)
                    for c in obs_df.columns:
                        obs_df[c] = obs_df[c].replace(code_labels)
                    index_cols = [code_labels.get(c, c) for c in index_cols]

                for i in index_cols:
                    if i not in obs_df.columns:
                        # Check if selector is one of the measures and drop all non-related
                        # Note: only ONE measure can be a selector. Secondary measures as selector will be ignored
                        obs_df = obs_df[obs_df['Measure'] == i]
                        if obs_df.empty:
                            raise ValueError(f"Selector column '{i}' not found in data table "
                                             f"{table} with columns {obs_df.columns.values}.")
                        else:
                            index_cols = {'Measure', 'Unit'}

                result_df = obs_df.pivot_table(index=index_cols,
                                               columns=[c for c in obs_df if c not in ['Value'] + list(index_cols)],
                                               values='Value')
        except (UnitCompatibilityError, SyntaxError) as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve data from OData{'4' if self._odata4 else '3'} request: {str(e)}")

        return result_df, code_labels

"""
    This module retrieves the data from a SQL query and returns a pivot table as a pandas DataFrame.
    This module is similar to the `odata_executor`, which does the same using the OData API.
"""
import duckdb
import numpy as np
import pandas as pd
import re
import warnings
from operator import itemgetter
from rdflib import Graph, SKOS, QB, DCTERMS as DCT
from sqlglot.errors import SqlglotError
from time import time
from typing import Set, FrozenSet, Tuple, List
from traceback import format_exception

from config import logging
from odata_graph import engine
from odata_graph.sparql_controller import QUDT
from s_expression.mapper import CodeLabelMapper
from s_expression import Table, Measure, Dimension, uri_to_code
from utils.custom_types import UnitCompatibilityError, FormatWarning


logger = logging.getLogger(__name__)


class DBExecutor(object):
    """
        Object for handling SQL queries on OData tables in DuckDB and Parquet files.

        :param tables: URIs and identifiers of tables to fetch
        :param measures: set of measure URIs to filter table on
        :param dims: set of dimension URIs to filter table on
        :param operator_name: name of the operator calling the DBExecutor. Relevant if unique actions are required
    """
    db = duckdb.connect()

    def __init__(self,
                 tables: List[Table],
                 measures: Set[Measure],
                 dims: Set[Tuple[Dimension, FrozenSet[Dimension]]],
                 operator_name: str = ""):
        self.tables = tables
        self.measures = measures
        self.dims = dims
        self.operator_name = operator_name

        self.graph = Graph()
        for table in tables:
            self.graph += engine.get_table_graph(table=table, include_time_geo_dims=True)

    def query_db(self,
                 query: str,
                 index_cols: FrozenSet = frozenset({'Measure', 'Unit'}),
                 simplified: bool = False,
                 friendly_labels: bool = True) -> Tuple[pd.DataFrame, CodeLabelMapper, float]:
        """
            Retrieve values from OData parquet files and create a pivot table using table, measure and dimension selectors.

            :param query: SQL query to execute on DuckDB
            :param index_cols: the columns to use as the MultiIndex for the output pivot tables.
                               Defaults to ['Measure', 'Unit']
            :param simplified: if executing simplified queries, measure columns/pivots are not mandatory and
                               units will be ignored.
            :param friendly_labels: map the OData codes in the output table to their friendly names.
                                    Set to False when wanting to JOIN the pivot tables at a later stage.
            :returns: tuple containing (a DataFrame pivot table, mapper for codes to labels, SQL execution time)
        """
        # Get friendly names of codes
        concept_labels = dict(self.graph.subject_objects(SKOS.prefLabel))
        code_labels = CodeLabelMapper(
            # MSR/DIM-Concept mapping
            {str(obs).split(Measure.rdf_ns if obs in Measure.rdf_ns else Dimension.rdf_ns)[-1]: str(concept_labels[concept])
             for obs, concept in self.graph.subject_objects(QB.concept)} |
            # Table title
            {str(table).split(Table.rdf_ns)[-1]: str(title)
             for table, title in self.graph.subject_objects(DCT.title)}
        )

        try:
            # Get the SQL query result
            t0 = time()
            observations = duckdb.sql(query)
            execution_time = time() - t0

            result_df = observations.df()
            if len(observations) > 0 and not result_df.dropna(how='all', axis=1).empty:
                obs_df = result_df.copy()
                obs_df.dropna(how='all', axis=1, inplace=True)

                # Drop all duplicate columns (can happen when performing JOINS)
                obs_df = obs_df[[c for c in obs_df.columns if not ((match := re.match(r'^(.*)_(\d{,2})$', c))
                                                                   and match.group(1) in obs_df.columns)]]
                obs_df.rename({'Dimension_Measure': 'Measure'}, axis=1, inplace=True)  # Only relevant for JOIN and PROP results

                if not simplified:
                    # Validate and add the relevant units for all measures
                    if self.operator_name in ['SUM', 'AVG', 'MIN', 'MAX']:
                        # TODO: check if this doesn't throw errors when summing over dimensions instead of units.
                        #  can we even check this?
                        engine.validate_msr_unit_compatibility(self.measures, allow_different_scaling=False)
                    units = {uri_to_code(m): str(u) for m, u in self.graph.subject_objects(QUDT.unitOfSystem)}
                    if 'Measure' in obs_df.columns:
                        obs_df['Unit'] = obs_df['Measure'].map(units)
                        obs_df.loc[obs_df['Measure'].str.contains(', '), 'Unit'] = (  # also add units for aggregated measures
                            obs_df[obs_df['Measure'].str.contains(', ')]['Measure'].apply(
                                lambda msr: ', '.join([units[code] for code in msr.split(', ') if code in units])
                            )
                        )
                        obs_df['Unit'].replace({np.nan: ''}, inplace=True)

                        # Give a warning if the aggregated units don't align
                        if obs_df['Unit'].str.split(', ').apply(lambda l: len(set(l)) > 1).any():
                            logger.warning("Aggregation done over measures with different units."
                                           "Be sure to check the answer on correctness.")

                # Put aggregated dimensions into a MultiIndex. Note: pivoted dimensions are separated by underscores _
                # in DuckDB, but because our identifiers can also contain underscores, we have to do some trickery
                # to split them correctly. Assumes that a code contains of at least three characters. This can still
                # be prone to errors.
                obs_df.columns = obs_df.columns.str.split(r'_(?=[^_]{3,})', regex=True, expand=True)

                # Give the pivot cols the names of their respective dimensions. Not relevant for JOINs, as the dimension
                # coming from different tables won't be identical over al MultiIndex columns.
                if friendly_labels:
                    code_to_group = {}
                    for name, codes in dict(self.dims).items():
                        for code in codes:
                            code_to_group[str(code)] = str(name)

                    if isinstance(obs_df.columns, pd.MultiIndex) and (self.operator_name != 'JOIN' or 'JOIN' in query):
                        # Go, for the first relevant MultiIndex column through each item and assign as its name the relevant
                        # dimension group corresponding with the code. Doesn't make sense for JOINs, as the identifiers can
                        # come from completely different tables and dimension groups
                        col_names = []
                        for i, labeled_code in enumerate([c for c in obs_df.columns.values if
                                                          c[0] not in ['Measure', 'Unit'] and c[0] not in index_cols][0]):
                            col_names.append(code_labels.get(c := code_to_group.get(labeled_code, labeled_code), c))
                        obs_df.columns.names = col_names

                    # Translate the codes in the answer df to their readable labels
                    obs_df = obs_df.rename(columns=code_labels)
                    if isinstance(obs_df.columns, pd.MultiIndex):
                        cleaned_levels = [tuple('' if pd.isna(i) else i for i in c) for c in obs_df.columns]
                        obs_df.columns = pd.MultiIndex.from_tuples(cleaned_levels, names=obs_df.columns.names)

                    for c in obs_df.columns:
                        obs_df[c] = obs_df[c].replace(code_labels, regex=True)
                    index_cols = [code_labels.get(c, c) for c in index_cols]

                if not simplified:
                    try:
                        for i in index_cols:
                            dim_groups = map(str, map(itemgetter(0), self.dims))
                            if not (i in obs_df.columns or i in obs_df.columns.names or i in dim_groups):
                                # Check if selector is one of the measures and drop all non-related
                                # Note: only ONE measure can be a selector. Secondary measures as selector will be ignored
                                obs_df = obs_df[obs_df['Measure'] == i]
                                if obs_df.empty:
                                    warnings.warn(f"Selector column '{i}' not found in data table with columns "
                                                  f"{obs_df.columns.values}. Continuing without indexing.", FormatWarning)
                                    break
                                else:
                                    index_cols = {'Measure', 'Unit'}

                        if len(index_cols) > 1:
                            obs_df.index = pd.MultiIndex.from_tuples(map(tuple, obs_df[index_cols].values), names=index_cols)
                        else:
                            obs_df.index = pd.Index(obs_df[index_cols].values.flatten())
                        obs_df = obs_df.drop(index_cols, axis=1)
                    except KeyError as e:
                        warnings.warn(f"Indexing on columns that are not returned by the query: {e}. "
                                      f"Continuing without indexing.", FormatWarning)
                else:
                    if 'rnk' in obs_df:
                        obs_df = obs_df.drop(['rnk'], axis=1)  # Drop 'rnk' utility created for the MIN/MAX type queries

                result_df = obs_df
        except (UnitCompatibilityError, duckdb.IOException, duckdb.ParserException, SqlglotError, duckdb.BinderException) as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to process retrieved data: {format_exception(e)}")

        # As a last step, sort the columns. This is automatically done by the OData4 API, but not by SQL
        return result_df[sorted(result_df)], code_labels, execution_time

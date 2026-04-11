import logging
import os.path
import textwrap
from rdflib import Graph, ConjunctiveGraph, Namespace, Literal, URIRef
from SPARQLWrapper import SPARQLWrapper, JSON, RDFXML
from time import time
from typing import Union, Set, Dict

import config
from odata_graph.namespaces._QUDT_UNIT import QUDT_UNIT
from s_expression import Table, Measure, Dimension, uri_to_code
from utils.custom_types import UnitCompatibilityError

logger = logging.getLogger(__name__)

SCOT = Namespace("http://statistics.gov.scot/def/dimension/")  # Great scot!
QUDT = Namespace("http://qudt.org/schema/qudt/")


class SparqlEngine(object):
    def __init__(self, local: bool = True):
        """
            Object for handling SPARQL request to the graph. The graph can run either locally
            or through a GraphDB connection.

            :param local: use local graph file if available
        """
        self.DEFAULT_ENDPOINT = f"{config.GRAPH_DB_HOST}/repositories/{config.GRAPH_DB_REPO}"
        self.DEFAULT_INSERT_ENDPOINT = f"{self.DEFAULT_ENDPOINT}/statements"

        if local or config.IS_UNIT_TESTING:
            if not os.path.exists(config.GRAPH_FILE):
                raise FileNotFoundError(f"No local graph file found at location {config.GRAPH_FILE}.")

            t0 = time()
            logger.info(f"Local file found. Loading graph from {config.GRAPH_FILE}. "
                        f"Depending on size this can take a while.")
            self.graph = ConjunctiveGraph(identifier=f"http://{config.GRAPH_DB_REPO}", store="Oxigraph")
            self.graph.parse(config.GRAPH_FILE, format=config.GRAPH_FILE.split('.')[-1])

            self.local = True
            logger.info(f"SPARQL SparqlEngine is running from local file! ({time() - t0} s)")
        else:
            self.sparql = SPARQLWrapper(self.DEFAULT_ENDPOINT)
            self.sparql.setCredentials(config.GRAPH_DB_USERNAME, config.GRAPH_DB_PASSWORD)

            # Check whether the credentials are configured correctly. Raises an exception if not
            self.sparql.setQuery("ASK WHERE { ?s a ?o }")
            self.sparql.query()

            self.local = False
            logger.info("SPARQL SparqlEngine is running from GraphDB!")

    def select(self, query: str, endpoint: str = None) -> list:
        if self.local:
            result = self.graph.query(query)
            # Map the rdflib.query result to the same format as SPARQLWrapper.queryAndConvert
            return [{str(k): {
                'type': 'literal' if isinstance(v, Literal) else 'uri',
                'value': str(v)
            } for k, v in b.items()} for b in result.bindings]
        else:
            self.sparql.setQuery(query)
            self._set_method_get(endpoint, JSON)

            try:
                res = self.sparql.queryAndConvert()
                return res['results']['bindings']
            except Exception as e:
                raise RuntimeError(f"Failed to perform SELECT query: {e}")

    def construct(self, query: str, endpoint: str = None) -> Union[ConjunctiveGraph, Graph]:
        if self.local:
            result = self.graph.query(query)
            return result.graph
        else:
            self.sparql.setQuery(query)
            self._set_method_get(endpoint, RDFXML)

            try:
                g = self.sparql.queryAndConvert()  # subgraph following from candidate nodes
                return g
            except Exception as e:
                raise RuntimeError(f"Failed to perform CONSTRUCT query: {e}")

    def insert(self, query: str, uri: str, endpoint: str = None, verbose: bool = False):
        try:
            if self.local:
                self.graph.update(query)
            else:
                self.sparql.setQuery(query)
                self._set_method_post(endpoint, JSON)

            if verbose:
                logger.debug(f"Inserted item with uri {uri} into graph.")
        except Exception as e:
            logger.error(f"Failed to perform INSERT query: {e}")
            raise e

    def _set_method_get(self, endpoint: str, return_format: str):
        self.sparql.endpoint = endpoint if endpoint else self.DEFAULT_ENDPOINT
        self.sparql.setReturnFormat(return_format)
        self.sparql.method = 'GET'

    def _set_method_post(self, endpoint: str, return_format: str):
        self.sparql.updateEndpoint = endpoint if endpoint else self.DEFAULT_INSERT_ENDPOINT
        self.sparql.setReturnFormat(return_format)
        self.sparql.setMethod('POST')
        self.sparql.query()

    def explode_subgraph(self, nodes: list, table_cutoff: int = 5, verbose=False) -> Graph:
        """
            Construct a subgraph from the GraphDB based on a list of given table, measure and
            dimension nodes. The graph returned will contain all the measure/dimension-table
            relations and the corresponding metadata of and hierarchies between all nodes.

            :param nodes: list of URI's (table, measure or dimension) of nodes to explode
            :param table_cutoff: maximum number of tables to create full subgraphs for
            :param verbose: print the full parsed and executed query
            :returns: subgraph
        """
        explode_tables = [n for n in nodes if n in Table.rdf_ns][:table_cutoff]  # TODO: with 5+ tables exploding the subgraph takes a long time
        table_filter = ("?s IN (<" + '>, <'.join(explode_tables) + ">)") if explode_tables else ""
        explode_obs = [n for n in nodes if n in Measure.rdf_ns or n in Dimension.rdf_ns][:10]  # TODO: request header becomes too large with too many observation nodes. 'Use scroll api'
        obs_filter = ("?o IN (<" + '>, <'.join(explode_obs) + ">)") if explode_obs else ""

        query = textwrap.dedent(f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX scot: <http://statistics.gov.scot/def/dimension/>
            PREFIX qudt: <http://qudt.org/schema/qudt/>
    
            CONSTRUCT {{ 
                ?s ?p ?obs ;                                                            # return table-msr/dim triples
                   dct:title ?t_title .
                ?obs ?hierarchy ?group ;                                                # return dim hierarchy triples
                     ?has_type ?type ;                                                  # return optional TC/GC indication
                     qb:concept ?concept ;
                     ?has_cbs_unit ?cbs_unit ;                                          # return optional units for measures
                     ?has_qudt_unit ?qudt_unit .
                ?concept dct:isPartOf ?s ;                                              # return concepts linked to tables and msr/dims
                         skos:prefLabel ?obs_label ;
                         ?has_def ?obs_def .
            }} WHERE {{
                {{                                                                      # get all tables and their MSRs/DIMs
                    SELECT DISTINCT ?s WHERE {{
                        ?s qb:measure|qb:dimension ?o .
                        FILTER NOT EXISTS {{                                            # don't explode time and geo dims
                            VALUES ?dim {{'TimeDimension' 'GeoDimension'}}
                            ?o a ?dim .
                        }}
                        FILTER ({table_filter} {'||' if explode_tables and explode_obs else ''}     # select specific table relevant triples
                                {obs_filter}) .                                                     # select specific MSR/DIM triples
                    }}
                }}
                ?s ?p ?obs                                                              # explode!
                FILTER (?p = qb:measure || ?p = qb:dimension) .
                ?s dct:title ?t_title .                                                 # fetch titles and labels for all subjects
                ?obs qb:concept ?concept .                                              # labels for all msrs and dims
                ?concept skos:prefLabel ?obs_label ;
                         dct:isPartOf ?s .
                FILTER (?p = qb:measure || ?p = qb:dimension) .
                FILTER NOT EXISTS {{
                    VALUES ?type {{'TimeDimension' 'GeoDimension'}}
                    ?obs a ?type ;
                         skos:broader ?d .                                              # only get dim groups for time and geo dims
                }}
                OPTIONAL {{
                    ?obs ?hierarchy ?group                                              # get hierarchy between dimensions (broader)
                    FILTER (?hierarchy = skos:broader || ?hierarchy = skos:narrower)
                    FILTER NOT EXISTS {{                                                # don't get hierarchy for time and geo dims
                        VALUES ?type {{'TimeDimension' 'GeoDimension'}}
                        ?obs a ?type .
                    }}
                    FILTER EXISTS {{ 
                        ?s ?p ?group .                                                  # only get hierarchy for dimension relevant to tables in subgraph
                    }}
                }}
                OPTIONAL {{                                                             # get notion if dim group is TC or GC
                    VALUES ?type {{'TimeDimension' 'GeoDimension' 
                                   scot:Total scot:confidenceInterval}}
                    ?obs a ?type .
                    ?obs ?has_type ?type .
                }}
                OPTIONAL {{                                                             # get OData4 units of measure (non-standardised)
                    ?obs qudt:unitOfSystem ?cbs_unit .
                    ?obs ?has_cbs_unit ?cbs_unit .
                }}
                OPTIONAL {{                                                             # get units of measure
                    ?obs qudt:unit ?qudt_unit .
                    ?obs ?has_qudt_unit ?qudt_unit .
                }}
                OPTIONAL {{                                                             # get msr/dim definitions
                    ?concept skos:definition ?obs_def .
                    ?concept ?has_def ?obs_def .
                }}
            }}
        """)

        if verbose:
            logger.debug(query)

        return self.construct(query)

    def explode_subgraph_msr_dims_only(self, nodes: list, table_cutoff: int = 5, verbose: bool = False) -> Graph:
        """
            Construct a subgraph from the GraphDB based on a list of given table, measure and
            dimension nodes. The graph returned will contain all the measure/dimension-table
            relations. This function omits all hierarchical relations between measures and
            dimensions, labels and units! Use `explode_subgraph()` to obtain the full graph if
            needed.

            :param nodes: list of URI's (table, measure or dimension) of nodes to explode
            :param table_cutoff: maximum number of tables to create full subgraphs for
            :param verbose: print the full parsed and executed query
            :returns: subgraph
        """
        explode_tables = [n for n in nodes if n in Table.rdf_ns][:table_cutoff]
        table_filter = ("?s IN (<" + '>, <'.join(explode_tables) + ">)") if explode_tables else ""
        explode_obs = [n for n in nodes if n in Measure.rdf_ns or n in Dimension.rdf_ns][:20]
        obs_filter = ("?o IN (<" + '>, <'.join(explode_obs) + ">)") if explode_obs else ""

        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    
            CONSTRUCT {{
                ?s ?p ?obs .                                                            # return table-msr/dim triples
            }} WHERE {{
                {{                                                                      # get all tables and their MSRs/DIMs
                    SELECT DISTINCT ?s WHERE {{
                        ?s qb:measure|qb:dimension ?o .
                        FILTER NOT EXISTS {{                                            # don't explode time and geo dims
                            VALUES ?dim {{'TimeDimension' 'GeoDimension'}}
                            ?o a ?dim .
                        }}
                        # select specific table relevant triples and select specific MSR/DIM triples
                        FILTER ({table_filter} {'||' if explode_tables and explode_obs else ''}
                                {obs_filter}) .
                    }}
                }}
                ?s ?p ?obs                                                              # explode!
                FILTER (?p = qb:measure || ?p = qb:dimension)
                FILTER NOT EXISTS {{                                                    # don't return time/geo dims
                    VALUES ?type {{'TimeDimension' 'GeoDimension'}}
                    ?obs a ?type ;
                         skos:broader ?d .
                }}
            }}
        """)

        if verbose:
            logger.debug(textwrap.dedent(query))

        return self.construct(query)

    def get_table_graph(self, table: Table, include_time_geo_dims: bool = False) -> Graph:
        """
            Construct a subgraph from the GraphDB based a single given table. The graph
            returned will contain all the measure/dimension-table relations and the
            corresponding metadata of and hierarchies between all nodes.

            :param table: table node to construct graph for
            :param include_time_geo_dims: if false, skips the time and geo dim codes, which can become huge
            :returns: subgraph
        """
        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX scot: <http://statistics.gov.scot/def/dimension/>
            PREFIX qudt: <http://qudt.org/schema/qudt/>

            CONSTRUCT {{ 
                ?s ?p ?obs ;                                                            # return table-msr/dim triples
                   dct:title ?t_title .
                ?obs ?hierarchy ?group ;                                                # return dim hierarchy triples
                     ?has_type ?type ;                                                  # return optional TC/GC indication
                     qb:concept ?concept ;
                     ?has_cbs_unit ?cbs_unit ;                                          # return optional units for measures
                     ?has_qudt_unit ?qudt_unit .
                ?concept dct:isPartOf ?s ;                                              # return concepts linked to tables and msr/dims
                         skos:prefLabel ?obs_label ;
                         ?has_def ?obs_def .
            }} WHERE {{
                BIND (<{table.uri}> AS ?s) .
                ?s ?p ?obs
                FILTER (?p = qb:measure || ?p = qb:dimension) .
                {'''
                FILTER NOT EXISTS {
                    VALUES ?type {'TimeDimension' 'GeoDimension'}
                    ?obs a ?type ;
                         skos:broader ?d .                                              # only get dim groups for time and geo dims
                }
                ''' if not include_time_geo_dims else ''}
                ?s dct:title ?t_title .
                ?obs qb:concept ?concept .                                              # labels for all msrs and dims
                ?concept skos:prefLabel ?obs_label ;
                         dct:isPartOf ?s .
                OPTIONAL {{
                    ?obs ?hierarchy ?group                                              # get hierarchy between dimensions (broader)
                    FILTER (?hierarchy = skos:broader || ?hierarchy = skos:narrower)
                    {'''
                    FILTER NOT EXISTS {                                                 # don't get hierarchy for time and geo dims
                        VALUES ?type {'TimeDimension' 'GeoDimension'}
                        ?obs a ?type.
                    }
                    ''' if not include_time_geo_dims else ''}
                    FILTER EXISTS {{ 
                        ?s ?p ?group .                                                  # only get hierarchy for dimension relevant to tables in subgraph
                    }}
                }}
                OPTIONAL {{                                                             # get notion if dim group is TC or GC
                    VALUES ?type {{'TimeDimension' 'GeoDimension' 
                                   scot:Total scot:confidenceInterval}}
                    ?obs a ?type .
                    ?obs ?has_type ?type .
                }}
                OPTIONAL {{                                                             # get OData4 units of measure (non-standardised)
                    ?obs qudt:unitOfSystem ?cbs_unit .
                    ?obs ?has_cbs_unit ?cbs_unit .
                }}
                OPTIONAL {{                                                             # get units of measure
                    ?obs qudt:unit ?qudt_unit .
                    ?obs ?has_qudt_unit ?qudt_unit .
                }}
                OPTIONAL {{                                                             # get msr/dim definitions
                    ?concept skos:definition ?obs_def .
                    ?concept ?has_def ?obs_def .
                }}
            }}
        """)

        return self.construct(query)

    def get_table_geo_dims(self, table: Table) -> dict:
        """Get all the geo dimension codes (not groups!) of a table."""
        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            
            SELECT DISTINCT ?id ?prefLabel WHERE {{
                BIND (<{table.uri}> AS ?s) .
                ?s qb:dimension ?dim .
                ?dim dct:identifier ?id ;
                     qb:concept ?concept .
                ?concept dct:isPartOf ?s ;
                         skos:prefLabel ?prefLabel .
                FILTER (!STRSTARTS(?id, "BU") || !STRSTARTS(?id, "WK"))   # TODO: for now skip BU en WK dimension to avoid overflow
                FILTER EXISTS {{
                    ?dim a 'GeoDimension' ;
                         skos:broader ?d .
                }}
            }}
        """)

        res = self.select(query)
        return {d['id']['value']: d['prefLabel']['value'] for d in res}

    def get_table_time_dims(self, table: Table) -> dict:
        """Get all the time dimension codes (not groups!) of a table."""
        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT DISTINCT ?dim ?prefLabel WHERE {{ 
                BIND (<{table.uri}> AS ?s) .
                ?s qb:dimension ?dim .
                ?dim dct:identifier ?id ;
                     qb:concept ?concept .
                ?concept dct:isPartOf ?s ;
                         skos:prefLabel ?prefLabel .
                FILTER EXISTS {{
                    ?dim a 'TimeDimension' ;
                         skos:broader ?d .
                }}
            }}
        """)

        res = self.select(query)
        return {uri_to_code(d['dim']['value']): d['prefLabel']['value'] for d in res}

    def get_dimension_codes(self, table: Table, dim_id: str) -> dict:
        """
            Return all (code → label) pairs for a specific schema dimension of a table.
            Mirrors get_table_geo_dims() but filters by the dimension group URI instead of type,
            making it usable for any non-geo, non-time dimension.

            :param table: table to query
            :param dim_id: dimension property ID, e.g. 'BestemmingEnSeizoen'
            :returns: dict mapping code string to prefLabel, e.g. {'T001460': 'Totaal bestemmingen'}
        """
        dim_uri = f"{Dimension.rdf_ns}{dim_id}"
        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT DISTINCT ?id ?prefLabel WHERE {{
                BIND (<{table.uri}> AS ?s) .
                ?s qb:dimension ?dim .
                ?dim dct:identifier ?id ;
                     qb:concept ?concept ;
                     skos:broader <{dim_uri}> .
                ?concept dct:isPartOf ?s ;
                         skos:prefLabel ?prefLabel .
            }}
        """)

        res = self.select(query)
        return {d['id']['value']: d['prefLabel']['value'] for d in res}

    def get_node_text_props(self, node: URIRef) -> dict:
        query = (f"""
            SELECT * WHERE {{ 
                <{node}> ?p ?o .
                FILTER (lang(?o) = "nl")
            }}
        """)

        res = self.select(query)
        return {uri_to_code(d['p']['value']): d['o']['value'] for d in res}

    def validate_msr_unit_compatibility(self, measures: Set[Measure], allow_different_scaling: bool = True) -> Dict[str, Dict[str, str]]:
        """
            Validate whether an operation performed on the measures is allowed based on the measures' units.
            Throws an TypeError when the operation is now allowed due to a unit mismatch.

            :param measures: the set of measures with URIs to validate
            :param allow_different_scaling: allow different units with a different scaling unit (e.g. kilograms + grams).
                                            Requires checks on conversion multipliers as well. This is not implemented for
                                            SQL execution, but is available for S-expressions.
            :returns: dictionary containing unit information
        """
        query = (f"""
            PREFIX qudt: <http://qudt.org/schema/qudt/>
            PREFIX quantitykind: <http://qudt.org/vocab/quantitykind/>

            SELECT ?msr ?unit ?scalingOf ?multiplier ?is_dimensionless WHERE {{
                VALUES ?msr {{ <{'> <'.join([m.uri for m in measures])}> }}
                OPTIONAL {{
                    ?msr qudt:unit ?q_unit .
                    OPTIONAL {{
                        ?q_unit qudt:scalingOf ?scalingOf .
                    }}
                }}
                OPTIONAL {{ ?msr qudt:unitOfSystem ?cbs_unit . }}
                OPTIONAL {{ ?msr qudt:conversionMultiplier ?multiplier . }}
                BIND(COALESCE(?q_unit, ?cbs_unit) AS ?unit)
                BIND(EXISTS {{
                    VALUES ?kind {{ quantitykind:Dimensionless quantitykind:DimensionlessRatio }}
                    ?unit qudt:hasQuantityKind ?kind .
                }} as ?is_dimensionless)
                FILTER (BOUND(?unit) || BOUND(?multiplier))
            }}
        """)

        res = self.select(query)
        measure_units = {uri_to_code(d['msr']['value']): {
            'unit': (u := d.get('unit', {'value': None})['value']),  # combination qudt:unit and qudt:unitOfSystem; see SPARQL  above
            'scalingOf': d['scalingOf']['value'] if 'scalingOf' in d and d['scalingOf']['value'] not in ['None', None] else u,
            'multiplier': d.get('multiplier', {'value': None})['value'],
            'is_dimensionless': d.get('is_dimensionless', {'value': 'false'})['value'] == 'true',
        } for d in res}

        # 1. Check if any measure's unit dimension is dimensionless, except for qudt:Num, which should give a warning
        #    Units with a dimensionless vector (e.g. qudt:PERCENT) are non-aggregatable
        if len(measure_units) > 1 and any(d['is_dimensionless'] and d['unit'] != str(QUDT_UNIT.COUNT) for d in measure_units.values()):
            raise UnitCompatibilityError(
                f"Trying to aggregate over two or more measures with a dimensionless unit:\n{measure_units}"
            )

        # 2. Check if qudt:unit are present and equal for all measures
        #    If the units are equal for all measures, they are aggregatable
        units = [d['unit'] for d in measure_units.values()]
        if None not in units and len(set(units)) == 1:
            if str(QUDT_UNIT.COUNT) in units:
                logger.warning(f"Aggregating over measures with dimensionless qudt:Count as unit. "
                               f"This could lead to improper aggregations. Measures and their units:\n"
                               f"{measure_units}")
            # Check if conversion multipliers are equal when scaling is not allowed
            if not allow_different_scaling:
                multipliers = [d['multiplier'] for d in measure_units.values()]
                if len(set(multipliers)) > 1:
                    raise UnitCompatibilityError(
                        f"Trying to aggregate over multiple measures with different conversion multipliers:\n"
                        f"{measure_units}"
                    )
            return measure_units

        # 3. If not, check if the scaling unit (qudt:scalingOf) is equal for all measures
        #    If qudt:scalingOf is equal for all measures, they are aggregatable
        # FIXME: the qudt:scalingOf for these units themselves must be applied first in order to make them aggregatable
        scales = [d['scalingOf'] for d in measure_units.values()]
        if allow_different_scaling and None not in scales and len(set(scales)) == 1:
            return measure_units

        # Too bad. The measures are incompatible
        raise UnitCompatibilityError(f"Trying to aggregate over measures with a different unit: {measure_units}")

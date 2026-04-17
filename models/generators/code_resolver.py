from rapidfuzz import process, fuzz

from odata_graph.sparql_controller import SparqlEngine
from s_expression import Table, uri_to_code


class SKOSCodeResolver:
    """
    Resolves natural-language phrases in a question to CBS observation codes by
    querying SKOS concept schemes in the Knowledge Graph.

    Covers schema dimensions only (non-geo, non-time). Geo and time dimensions
    are already handled by the existing match_region() and extract_tc() helpers
    in BaseLLMGenerator._build_prompt().

    Inspired by ReForCE's Column Exploration approach (probing a data source to
    discover actual values for grounding SQL generation), adapted to the LOCuST
    Knowledge Graph setting: SPARQL queries on SKOS concept schemes replace SQL
    SELECT DISTINCT probes.
    """

    def __init__(self, engine: SparqlEngine, threshold: int = 80):
        """
        :param engine: SparqlEngine instance
        :param threshold: minimum rapidfuzz score (0–100) for a match to be accepted
        """
        self.engine = engine
        self.threshold = threshold

    def resolve(self, question: str, table_id: str) -> dict[str, str]:
        """
        Returns {dim_id: code} for schema dims whose labels fuzzy-match phrases
        in the question.

        Example: 'How many detached houses...' + table 83023NED
          → {'Woningkenmerken': 'ZW10320'}

        :param question: NL or EN question string
        :param table_id: CBS table ID, e.g. '83023NED'
        :returns: dict mapping dimension ID to the best-matching code
        """
        table = Table(table_id)
        schema_dim_ids = self._get_schema_dim_ids(table)

        resolved = {}
        for dim_id in schema_dim_ids:
            codes = self.engine.get_dimension_codes(table, dim_id)
            if not codes:
                continue
            match = self._fuzzy_match(question, codes)
            if match:
                resolved[dim_id] = match
        return resolved

    def _get_schema_dim_ids(self, table: Table) -> list[str]:
        """
        Return schema dimension property IDs for a table via direct SPARQL SELECT.

        Schema dims = qb:dimension group nodes that are NOT individual codes
        (i.e. have no skos:broader parent) and are NOT geo or time dimensions.
        Geo/time dims are excluded because BaseLLMGenerator already resolves them
        via match_region() / extract_tc() — injecting duplicate hints is noisy.
        Uses SELECT directly rather than parsing the CONSTRUCT graph, which avoids
        issues with Oxigraph returning empty CONSTRUCT results for some optional patterns.
        """
        query = f"""
            PREFIX qb:   <http://purl.org/linked-data/cube#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT DISTINCT ?dim WHERE {{
                <{table.uri}> qb:dimension ?dim .
                FILTER NOT EXISTS {{ ?dim skos:broader ?any . }}
                FILTER NOT EXISTS {{ ?code a 'GeoDimension' ; skos:broader ?dim . }}
                FILTER NOT EXISTS {{ ?code a 'TimeDimension' ; skos:broader ?dim . }}
            }}
        """
        res = self.engine.select(query)
        return [uri_to_code(r['dim']['value']) for r in res]

    def _fuzzy_match(self, question: str, code_label_map: dict) -> str | None:
        """
        Find the best-matching code whose label appears in the question.

        :param question: question string to search in
        :param code_label_map: {code: label} dict from get_dimension_codes()
        :returns: the code of the best match, or None if below threshold
        """
        labels = list(code_label_map.values())
        codes = list(code_label_map.keys())
        result = process.extractOne(question, labels, scorer=fuzz.partial_ratio)
        if result and result[1] >= self.threshold:
            return codes[result[2]]
        return None

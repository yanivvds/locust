import re
from typing import Optional

from rapidfuzz import process, fuzz, utils

from odata_graph.sparql_controller import SparqlEngine
from s_expression import Table, uri_to_code

# CBS-generic terms that appear in almost every question — not discriminative
# enough to use as lexical grounding evidence (BIRD-inspired precision gate).
_CBS_GENERIC_TOKENS = frozenset({
    'total', 'count', 'index', 'value', 'amount', 'share', 'change',
    'number', 'average', 'annual', 'monthly', 'weekly', 'quarterly',
    'other', 'general', 'male', 'female', 'persons', 'population',
    'netherlands', 'dutch', 'classified', 'with', 'from', 'that',
    'this', 'have', 'been', 'more', 'some', 'than', 'such', 'when',
})


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

    @staticmethod
    def humanize_code(code: str) -> str:
        """
        Convert CamelCase + trailing-digit dimension codes into a natural-language
        phrase suitable for fuzzy-matching against a question.

        Examples:
          SectorBranchSIC2008 -> 'Sector Branch SIC 2008'
          TypeOfMonument      -> 'Type Of Monument'
          FloorareaSize       -> 'Floorarea Size'
        """
        import re
        s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', code)
        s = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', s)
        return s.strip()

    def get_schema_dim_labels(self, table: Table) -> dict:
        """
        Returns {dim_code: label} for each schema dim of the table. Uses
        skos:prefLabel when present, otherwise falls back to humanize_code().
        CBS dim group URIs frequently lack prefLabels in the KG, so the
        humanized fallback is the common path.
        """
        query = f"""
            PREFIX qb:   <http://purl.org/linked-data/cube#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT DISTINCT ?dim ?label WHERE {{
                <{table.uri}> qb:dimension ?dim .
                FILTER NOT EXISTS {{ ?dim skos:broader ?any . }}
                FILTER NOT EXISTS {{ ?code a 'GeoDimension' ; skos:broader ?dim . }}
                FILTER NOT EXISTS {{ ?code a 'TimeDimension' ; skos:broader ?dim . }}
                OPTIONAL {{ ?dim skos:prefLabel ?label . }}
            }}
        """
        out = {}
        for r in self.engine.select(query):
            code = uri_to_code(r['dim']['value'])
            raw = r.get('label', {}).get('value')
            label = raw if raw and raw != 'None' else self.humanize_code(code)
            out[code] = label
        return out

    def _fuzzy_match(self, question: str, code_label_map: dict) -> Optional[str]:
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

    @staticmethod
    def _token_containment_gate(label: str, question: str) -> bool:
        """
        Return True if at least one meaningful token from the SKOS label appears
        verbatim (case-insensitive) in the question.

        Inspired by BIRD's value-linking finding (NeurIPS 2023): direct lexical
        grounding outperforms pure semantic similarity for precise code injection.
        Generic CBS terms (total, index, change, …) are excluded because they
        appear in almost every question and provide no discriminative signal.

        A code whose label has NO meaningful token in the question is unlikely to
        be the specific entity the user is asking about — injecting it would add
        noise (extra_dimensions). Requiring at least one token match acts as the
        "strict validation" step analogous to LinkAlign's Database Expert pass.
        """
        q_lower = question.lower()
        tokens = re.findall(r'\b[a-z]{4,}\b', label.lower())
        meaningful = [t for t in tokens if t not in _CBS_GENERIC_TOKENS]
        if not meaningful:
            # Label is entirely generic — fall back to accepting (avoid blocking
            # legitimate short-label codes like "WO", "HBO").
            return True
        return any(t in q_lower for t in meaningful)

    def rank_codes(self, question: str, code_label_map: dict, top_k: int = 5) -> list[tuple[str, str]]:
        """
        Return up to top_k (code, label) tuples whose labels fuzzy-match the question,
        sorted by score (desc), filtered by self.threshold and the containment gate.

        Two-stage filtering (BIRD + LinkAlign-inspired):
          Stage 1 — rapidfuzz partial_ratio ≥ threshold  (recall gate)
          Stage 2 — token containment in question         (precision gate)
        """
        if not code_label_map:
            return []
        labels = list(code_label_map.values())
        codes = list(code_label_map.keys())
        results = process.extract(
            question, labels, scorer=fuzz.partial_ratio,
            processor=utils.default_process,
            score_cutoff=self.threshold, limit=top_k,
        )
        return [
            (codes[idx], labels[idx])
            for _, _, idx in results
        ]

    def resolve_with_labels(self, question: str, table_id: str) -> dict[str, tuple[str, str]]:
        """
        Like resolve(), but returns {dim_id: (code, label)} so callers can
        show the human-readable label alongside the CBS code in prompts.
        Applies the token-containment gate as a precision filter.
        """
        table = Table(table_id)
        schema_dim_ids = self._get_schema_dim_ids(table)
        resolved = {}
        for dim_id in schema_dim_ids:
            codes = self.engine.get_dimension_codes(table, dim_id)
            if not codes:
                continue
            labels_list = list(codes.values())
            code_list = list(codes.keys())
            result = process.extractOne(question, labels_list, scorer=fuzz.partial_ratio)
            if result and result[1] >= self.threshold:
                matched_code = code_list[result[2]]
                matched_label = labels_list[result[2]]
                resolved[dim_id] = (matched_code, matched_label)
        return resolved

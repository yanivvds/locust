"""
KG-Enriched BM25 Retriever.

Same keyword-matching approach as BM25Retriever, but each table's index body
is enriched with the preferred and alternative labels of all its dimensions
and measures from the knowledge graph.

Hypothesis: BM25 fails when question terms (e.g. "transportation sector") only
appear in dimension labels, not in the table title. Appending those labels to
the table body should fix this.

Design differences from the base BM25Retriever:
  - Single flat BM25 index over enriched table bodies (no separate dim/msr indices)
  - No multi-component score combination — just table-level BM25
  - Only prefLabel and altLabel are appended (short, informative); not definitions
    or long descriptions (too noisy)
  - Time and geo dimensions excluded (same as base BM25)
  - Cache stored separately so both retrievers can coexist
"""
import itertools
import json
import nltk
import os
import string
from collections import OrderedDict
from operator import itemgetter
from tqdm import tqdm
from typing import Dict, List

from rank_bm25 import BM25Plus

import config
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.bm25_retriever import process_text
from odata_graph import engine

CACHE_PATH = f"{config.PATH_DIR_DATA}/bm25_retriever/kg_enriched_bm25_{config.GRAPH_DB_REPO}_node_labels.json"


class KGEnrichedBM25Retriever(BaseRetriever):
    """
    BM25 retriever where each table body also contains all dimension and
    measure labels from the knowledge graph.
    """
    index: BM25Plus
    table_ids: List[str]

    def __init__(self, enrich: str = 'True'):
        """
        :param enrich: 'True' (default) adds KG dimension/measure labels to each table body.
                       'False' uses only title/abstract/description — same flat scoring,
                       no KG enrichment. Use this for the ablation that isolates the
                       effect of enrichment from the effect of flat scoring.
        """
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        super().__init__()

        self.enrich = enrich != 'False'
        enriched_tables = get_enriched_table_labels(enrich=self.enrich)
        self.table_ids = list(enriched_tables.keys())
        corpus = [enriched_tables[t]['body'] for t in self.table_ids]
        tokenized_corpus = [doc.split() for doc in corpus]

        label = "KG-enriched" if self.enrich else "flat (no enrichment)"
        print(f"Creating {label} BM25+ index for tables...")
        self.index = BM25Plus(tokenized_corpus)

    def retrieve_tables(self, question: str, k: int = 5, **kwargs) -> OrderedDict:
        """
        Return the top-k tables sorted by BM25 score over enriched bodies.

        :param question: natural language question
        :param k: number of tables to return
        :return: OrderedDict {table_id: {'score': float, 'dimensions': {}, 'measures': {}}}
        """
        processed = process_text(question)
        scores = self.index.get_scores(processed.split())

        ranked = sorted(
            zip(self.table_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = OrderedDict()
        for table_id, score in ranked[:k]:
            results[table_id] = {'score': score, 'dimensions': {}, 'measures': {}}
        return results


def get_enriched_table_labels(enrich: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Build (or load from cache) a dict of {table_id: {body, type}} where
    body = table title/abstract/description, optionally enriched with all
    dimension/measure prefLabels and altLabels from the KG (time/geo excluded).

    :param enrich: if True, append KG dimension/measure labels to each table body.
                   if False, use only title/abstract/description (ablation baseline).
    """
    cache_path = CACHE_PATH if enrich else CACHE_PATH.replace('kg_enriched', 'flat')

    if os.path.isfile(cache_path):
        label = "enriched" if enrich else "flat (ablation)"
        print(f"Loading {label} node labels from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    # ── Step 1: fetch all table titles ───────────────────────────────────────
    table_query = """
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dct: <http://purl.org/dc/terms/>

        SELECT DISTINCT ?id ?label WHERE {
            ?s a dcat:Dataset .
            ?s dct:identifier ?id .
            OPTIONAL { ?s dct:title|dct:abstract|dct:description ?label }
        }
    """
    try:
        rows = engine.select(table_query)
        props = [(r['id']['value'], (r.get('label', False) or {'value': ''})['value']) for r in rows]
        table_it = itertools.groupby(props, itemgetter(0))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch table IDs: {e}")

    tables: Dict[str, Dict] = {}
    all_table_ids = list({p[0] for p in props})
    desc = "Building enriched table bodies" if enrich else "Building flat table bodies (ablation)"

    for table_id, table_props in tqdm(
        table_it,
        total=len(all_table_ids),
        desc=desc,
        bar_format=config.TQDM_BAR_FMT
    ):
        # Base body: table title / abstract / description
        base_text = ' '.join(
            f"{p[1].strip()}{'.' if p[1] and p[1][-1] != '.' else ''}"
            for p in table_props if p[1]
        )

        dim_labels = ''
        if enrich:
            # ── Step 2: fetch dimension and measure labels for this table ─────
            dim_msr_query = f"""
                PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                PREFIX dct: <http://purl.org/dc/terms/>
                PREFIX qb: <http://purl.org/linked-data/cube#>

                SELECT ?label WHERE {{
                    ?table dct:identifier "{table_id}" .
                    ?table qb:measure|qb:dimension ?node .
                    FILTER NOT EXISTS {{
                        VALUES ?timegeo {{'TimeDimension' 'GeoDimension'}}
                        ?node a ?timegeo .
                    }}
                    ?node qb:concept ?concept .
                    ?concept dct:isPartOf ?table .
                    ?concept skos:prefLabel|skos:altLabel ?label .
                }}
            """
            try:
                dim_rows = engine.select(dim_msr_query)
                dim_labels = ' '.join(
                    r['label']['value'].strip() for r in dim_rows if r.get('label')
                )
            except Exception as e:
                print(f"  Warning: could not fetch dim/msr labels for {table_id}: {e}")

        full_text = f"{base_text} {dim_labels}".strip()
        tables[table_id] = {
            'body': process_text(full_text),
            'type': 'table'
        }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(tables, f)
    print(f"Saved node labels to {cache_path}")
    return tables

"""
LLM Reranker with Enriched Table Metadata (C3-a).

Extends LLMTableReranker by injecting dimension and measure labels into the
reranking prompt. The base class sends only title + description (which is useless
boilerplate "CONTENTS 1. General information..."). This variant fetches the actual
schema via SPARQL: dimension concept names (e.g. "Regions", "Period") and measure
names (e.g. "Declared insolvencies", "Of which enterprises").

Hypothesis: richer schema context helps the LLM distinguish near-duplicate tables
(e.g. "Bankruptcies; enterprises, regions" vs "Bankruptcies; natural persons, regions")
by exposing measure-level differences that titles alone cannot convey.

Usage (C3-a — ColBERT first stage, OpenAI):
    env PYTHONPATH=. LANGUAGE=en OPENAI_API_KEY=<key> \\
        python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/llm_enriched_reranker.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --backend openai \\
        --model gpt-5.4-mini \\
        --retrieval_k 20 \\
        --output_path evaluation/results/table_retrieval/c3a_openai_colbert_enriched.json
"""
import os
import re

import config
from logs import logging
from models.retrievers.llm_reranker import LLMTableReranker
from odata_graph import engine
from s_expression import Table

logger = logging.getLogger(__name__)


class LLMEnrichedReranker(LLMTableReranker):
    """ColBERT first stage + LLM reranker with enriched schema prompt (C3-a).

    Identical to LLMTableReranker except the prompt includes dimension concept labels
    and measure names fetched via SPARQL, replacing the useless dct:description field.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override with the enriched prompt template
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            f"{config.LANGUAGE}_enriched_reranker_prompt.txt",
        )
        with open(prompt_path) as f:
            prompt_text = f.read()
        parts = re.split(r'\[SYSTEM\]|\[USER\]', prompt_text)
        self.system_prompt = parts[1].strip()
        self.user_prompt_template = parts[2].strip()

    def retrieve_tables(self, query: str, k: int):
        from collections import OrderedDict
        pool_k = max(self.retrieval_k, k)
        candidates = self.colbert.retrieve_tables(query, k=pool_k)
        table_ids = list(candidates.keys())

        try:
            table_titles = engine.get_table_titles(
                [str(Table(tid).uri) for tid in table_ids]
            )
        except Exception:
            table_titles = {}

        try:
            schema_labels = engine.get_table_schema_labels(
                [str(Table(tid).uri) for tid in table_ids]
            )
        except Exception:
            logger.warning("get_table_schema_labels failed; falling back to titles only")
            schema_labels = {}

        table_info = self._build_enriched_table_info(table_ids, table_titles, schema_labels)
        user_prompt = self.user_prompt_template.format(
            question=query,
            table_info=table_info,
            k=k,
        )

        raw_response = self._call_llm(user_prompt)

        valid_ids_upper = {tid.upper(): tid for tid in table_ids}
        ordered_ids = self._parse_response(raw_response, valid_ids_upper, table_ids)

        return self._build_result(ordered_ids, k)

    @staticmethod
    def _build_enriched_table_info(table_ids: list, titles: dict, schema: dict) -> str:
        """Build a richer table listing with dimension and measure labels.

        Format per table:
            1. 82522ENG: Bankruptcies; enterprises, regions
               Measures: Declared insolvencies, Of which enterprises
               Dimensions: Regions, Period

        :param table_ids: ColBERT-ordered candidate IDs
        :param titles: {table_id: {'title': ..., 'description': ...}} from get_table_titles
        :param schema: {table_id: {'measures': [...], 'dimensions': [...]}} from get_table_schema_labels
        """
        lines = []
        for i, tid in enumerate(table_ids):
            meta = titles.get(tid, {})
            title = meta.get('title', tid)
            line = f"{i + 1}. {tid}: {title}"

            sch = schema.get(tid, {})
            measures = sch.get('measures', [])
            dimensions = sch.get('dimensions', [])

            if measures:
                line += f"\n   Measures: {', '.join(measures[:8])}"
            if dimensions:
                line += f"\n   Dimensions: {', '.join(dimensions[:6])}"

            lines.append(line)
        return '\n'.join(lines)

"""
LLM Reranker with Full Schema Enrichment: measures, dimensions, and time coverage (C3-c).

Extends LLMEnrichedReranker by pre-loading three caches at init:
  1. time_coverage  — {table_id: 'YYYY–YYYY'} from get_all_table_time_coverage()
  2. schema_cache   — {table_id: {measures, dimensions}} from get_all_table_schema_labels()
  3. title_cache    — {table_id: {title, description}} from get_all_table_titles()

All caches are built once; per-question overhead is O(1) dict lookups, avoiding
the per-question SPARQL overhead of the non-cached enriched variant.

Hypothesis: temporal context resolves ColBERT failures where the question specifies
a year or period only some candidates cover. Measures help distinguish near-duplicate
tables (e.g. income vs. poverty vs. Gini tables that all show "income distribution").

Usage (C3-c — ColBERT first stage, OpenAI):
    env PYTHONPATH=. LANGUAGE=en OPENAI_API_KEY=<key> \\
        python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/llm_fully_enriched_reranker.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --backend openai \\
        --model gpt-5.4-mini \\
        --retrieval_k 20 \\
        --output_path evaluation/results/table_retrieval/c3c_openai_colbert_full.json
"""
import os
import re

import config
from logs import logging
from models.retrievers.llm_reranker import LLMTableReranker
from odata_graph import engine

logger = logging.getLogger(__name__)


class LLMFullyEnrichedReranker(LLMTableReranker):
    """ColBERT first stage + LLM reranker with full schema + time coverage (C3-c).

    All table metadata is pre-loaded at init; per-question cost is just the LLM call.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override with fully enriched prompt
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            f"{config.LANGUAGE}_fully_enriched_reranker_prompt.txt",
        )
        with open(prompt_path) as f:
            prompt_text = f.read()
        parts = re.split(r'\[SYSTEM\]|\[USER\]', prompt_text)
        self.system_prompt = parts[1].strip()
        self.user_prompt_template = parts[2].strip()

        logger.info("Pre-loading all table metadata caches...")
        self.title_cache = engine.get_all_table_titles()
        self.schema_cache = engine.get_all_table_schema_labels()
        self.time_coverage = engine.get_all_table_time_coverage()
        logger.info(
            f"Caches ready: {len(self.title_cache)} titles, "
            f"{len(self.schema_cache)} schemas, "
            f"{len(self.time_coverage)} time ranges"
        )

    def retrieve_tables(self, query: str, k: int):
        from collections import OrderedDict
        pool_k = max(self.retrieval_k, k)
        candidates = self.colbert.retrieve_tables(query, k=pool_k)
        table_ids = list(candidates.keys())

        table_info = self._build_fully_enriched_table_info(
            table_ids, self.title_cache, self.schema_cache, self.time_coverage
        )
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
    def _build_fully_enriched_table_info(
        table_ids: list,
        titles: dict,
        schema: dict,
        time_coverage: dict,
    ) -> str:
        """Build table listing with title, time coverage, measures, and dimensions.

        Format per table:
            1. 82522ENG: Bankruptcies; enterprises, regions
               Period: 2009–2025
               Measures: Pronounced bankruptcies
               Dimensions: Regions, Type of bankruptcy
        """
        lines = []
        for i, tid in enumerate(table_ids):
            meta = titles.get(tid, {})
            title = meta.get('title', tid)
            line = f"{i + 1}. {tid}: {title}"

            time_cov = time_coverage.get(tid, '')
            if time_cov:
                line += f"\n   Period: {time_cov}"

            sch = schema.get(tid, {})
            measures = sch.get('measures', [])
            # Exclude "Periods"/"Perioden" from dimensions — covered by Period line
            dimensions = [d for d in sch.get('dimensions', [])
                          if d.lower() not in ('periods', 'perioden')]
            if measures:
                line += f"\n   Measures: {', '.join(measures[:8])}"
            if dimensions:
                line += f"\n   Dimensions: {', '.join(dimensions[:6])}"

            lines.append(line)
        return '\n'.join(lines)

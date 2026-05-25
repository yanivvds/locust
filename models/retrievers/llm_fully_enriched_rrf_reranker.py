"""
LLM Reranker with Full Schema + Time Coverage + RRF First Stage (C3-d).

Combines the fully enriched prompt from LLMFullyEnrichedReranker (pre-loaded
title, schema, and time coverage caches) with the B5-a RRF first stage.

Usage (C3-d — RRF first stage, OpenAI):
    env PYTHONPATH=. LANGUAGE=en OPENAI_API_KEY=<key> \\
        python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/llm_fully_enriched_rrf_reranker.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --backend openai \\
        --model gpt-5.4-mini \\
        --retrieval_k 20 \\
        --output_path evaluation/results/table_retrieval/c3d_openai_rrf_full.json
"""
import os
import re

import config
from logs import logging
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.hybrid_rrf_retriever import HybridRRFRetriever
from models.retrievers.llm_fully_enriched_reranker import LLMFullyEnrichedReranker
from openai import OpenAI
from odata_graph import engine

logger = logging.getLogger(__name__)


class LLMFullyEnrichedRRFReranker(LLMFullyEnrichedReranker):
    """RRF first stage + LLM reranker with full schema + time coverage (C3-d)."""

    def __init__(
        self,
        checkpoint: str,
        model: str = 'google/gemma-4-31B-it',
        base_url: str = 'http://localhost:8000/v1',
        retrieval_k: int = 20,
        rrf_k: int = 60,
        max_tokens: int = 256,
        collection_variant: str = None,
        enable_thinking: bool = False,
        backend: str = 'vllm',
    ):
        BaseRetriever.__init__(self)

        self.retrieval_k = int(retrieval_k)
        self.max_tokens = int(max_tokens)
        if collection_variant is True:
            collection_variant = None
        self.collection_variant = collection_variant
        self.enable_thinking = enable_thinking is True or str(enable_thinking).lower() == 'true'
        self.backend = str(backend).lower()

        self.colbert = HybridRRFRetriever(
            checkpoint=checkpoint,
            collection_variant=collection_variant or 'kg_units',
            retrieval_k=int(retrieval_k),
            rrf_k=int(rrf_k),
        )

        if self.backend == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required when backend='openai'")
            self.ml_client = OpenAI(api_key=api_key, timeout=120)
        else:
            self.ml_client = OpenAI(api_key='vllm', base_url=base_url, timeout=120)

        self.model_name = model

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

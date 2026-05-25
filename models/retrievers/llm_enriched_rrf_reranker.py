"""
LLM Reranker with Enriched Schema Prompt + RRF First Stage (C3-b).

Combines the enriched schema prompt from LLMEnrichedReranker with the
B5-a RRF first stage from LLMRRFReranker. This is the RRF analogue of C3-a.

Expected: tests whether enriched context resolves the calibration mismatch
seen in C2-b (RRF + plain gpt-5.4-mini gave acc@1=0.586, −0.4pp vs B5-a).

Usage (C3-b — RRF first stage, OpenAI):
    env PYTHONPATH=. LANGUAGE=en OPENAI_API_KEY=<key> \\
        python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/llm_enriched_rrf_reranker.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --backend openai \\
        --model gpt-5.4-mini \\
        --retrieval_k 20 \\
        --output_path evaluation/results/table_retrieval/c3b_openai_rrf_enriched.json
"""
import os
import re

import config
from logs import logging
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.hybrid_rrf_retriever import HybridRRFRetriever
from models.retrievers.llm_enriched_reranker import LLMEnrichedReranker
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMEnrichedRRFReranker(LLMEnrichedReranker):
    """RRF first stage + LLM reranker with enriched schema prompt (C3-b).

    Swaps the ColBERT first stage for HybridRRFRetriever (B5-a), giving a
    higher recall ceiling (0.951 vs 0.915) before the enriched LLM rerank step.
    """

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
        # Call grandparent to avoid LLMTableReranker's ColBERT init
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
            f"{config.LANGUAGE}_enriched_reranker_prompt.txt",
        )
        with open(prompt_path) as f:
            prompt_text = f.read()
        parts = re.split(r'\[SYSTEM\]|\[USER\]', prompt_text)
        self.system_prompt = parts[1].strip()
        self.user_prompt_template = parts[2].strip()

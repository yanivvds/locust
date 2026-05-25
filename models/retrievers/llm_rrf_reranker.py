"""
LLM Reranker with RRF First Stage — B5-a (BM25 + ColBERT) + LLM reranker.

Replaces the ColBERT-only first stage in LLMTableReranker with HybridRRFRetriever,
giving a higher recall ceiling (0.951 vs 0.912) before the LLM reranking step.

This is experiment S2. Expected: acc@1 > 0.671 (ColBERT+LLM baseline) because
the RRF pool contains the gold table in more cases before the LLM sees it.

Usage (Snellius, vLLM server must be running):
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/llm_rrf_reranker.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --model google/gemma-4-31B-it \\
        --base_url http://localhost:8000/v1 \\
        --retrieval_k 20 \\
        --rrf_k 60 \\
        --output_path evaluation/results/table_retrieval/s2_rrf_llm_reranker.json

Usage (OpenAI API, local — C2-b):
    env PYTHONPATH=. LANGUAGE=en OPENAI_API_KEY=<key> \\
        python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/llm_rrf_reranker.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --backend openai \\
        --model gpt-5.4-mini \\
        --retrieval_k 20 \\
        --output_path evaluation/results/table_retrieval/c2b_openai_rrf.json
"""
import os
import time
from collections import OrderedDict

from openai import OpenAI

import config
from logs import logging
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.hybrid_rrf_retriever import HybridRRFRetriever
from models.retrievers.llm_reranker import LLMTableReranker
from odata_graph import engine
from s_expression import Table

logger = logging.getLogger(__name__)


class LLMRRFReranker(LLMTableReranker):
    """B5-a RRF first stage + LLM reranker (S2 / C2-b).

    Identical to LLMTableReranker except the first stage is HybridRRFRetriever
    (BM25 + ColBERT via RRF) rather than ColBERT alone. The LLM reranking
    logic, prompt, and output parsing are unchanged.
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
        # Call grandparent BaseRetriever.__init__ to avoid LLMTableReranker's ColBERT init
        BaseRetriever.__init__(self)

        self.retrieval_k = int(retrieval_k)
        self.max_tokens = int(max_tokens)
        if collection_variant is True:
            collection_variant = None
        self.collection_variant = collection_variant
        self.enable_thinking = enable_thinking is True or str(enable_thinking).lower() == 'true'
        self.backend = str(backend).lower()

        # Replace colbert-only first stage with RRF
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

        import re as _re
        import os as _os
        prompt_path = _os.path.join(
            _os.path.dirname(__file__),
            f"{config.LANGUAGE}_reranker_prompt.txt",
        )
        with open(prompt_path) as f:
            prompt_text = f.read()

        parts = _re.split(r'\[SYSTEM\]|\[USER\]', prompt_text)
        self.system_prompt = parts[1].strip()
        self.user_prompt_template = parts[2].strip()

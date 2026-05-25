"""
LLM Reranker with SKOS Escape-Pool Injection (stacking experiment B7+LLM).

Thin subclass of LLMTableReranker that replaces the inner ColBERT retriever
with SKOSEscapeRetriever. The LLM reranker then operates on an augmented pool
that includes SKOS-concept-matched tables which ColBERT may have missed.

See models/retrievers/skos_escape_retriever.py (B7) and
    models/retrievers/llm_reranker.py (LLM reranker baseline).

Usage (Snellius, vLLM server must be running):
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/llm_reranker_escape.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --model google/gemma-4-31B-it \\
        --base_url http://localhost:8000/v1 \\
        --retrieval_k 20 \\
        --min_token_match 2 \\
        --output_path evaluation/results/table_retrieval/s3_llm_escape_min2.json
"""
from models.retrievers.llm_reranker import LLMTableReranker
from models.retrievers.skos_escape_retriever import SKOSEscapeRetriever


class LLMRerankerWithEscape(LLMTableReranker):
    """LLM listwise reranker whose first-stage retriever is SKOSEscapeRetriever.

    The LLM receives a pool that is the union of ColBERT's top-retrieval_k
    candidates and any SKOS-concept-matched tables not already in that pool.
    """

    def __init__(
        self,
        checkpoint: str,
        model: str = 'google/gemma-4-31B-it',
        base_url: str = 'http://localhost:8000/v1',
        retrieval_k: int = 20,
        max_tokens: int = 256,
        collection_variant: str = None,
        enable_thinking: bool = False,
        min_token_match: int = 2,
    ):
        """
        :param min_token_match: minimum SKOS token matches to inject a table
                                into the pool (passed to SKOSEscapeRetriever).
                                See B7 experiment matrix for values tested.
        All other parameters are forwarded to LLMTableReranker.
        """
        super().__init__(
            checkpoint=checkpoint,
            model=model,
            base_url=base_url,
            retrieval_k=retrieval_k,
            max_tokens=max_tokens,
            collection_variant=collection_variant,
            enable_thinking=enable_thinking,
        )
        # Replace the plain ColBERT inner retriever with the SKOS-escape variant.
        # LLMTableReranker.retrieve_tables calls self.colbert.retrieve_tables()
        # so this substitution is transparent to the parent class.
        self.colbert = SKOSEscapeRetriever(
            checkpoint=checkpoint,
            collection_variant=collection_variant or 'kg_units',
            colbert_k=retrieval_k,
            min_token_match=int(min_token_match),
        )

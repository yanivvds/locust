"""
Hybrid RRF + KG Overlap Reranker — BM25 + ColBERT via RRF, then KG schema-overlap reranking.

Stacks the B5-a RRF retriever (HybridRRFRetriever) with the KG schema-overlap
reranker. This is experiment S1.

IMPORTANT — alpha scale:
    The ColBERT-only KG reranker used alpha=22.5, tuned for raw ColBERT MaxSim scores
    (~30-50). RRF scores are much smaller (~0.013-0.033). This class normalizes the
    RRF pool scores to [0,1] before adding the KG overlap bonus, so alpha is
    scale-invariant: alpha=1.0 means "max KG bonus equals the max pool score."

    Recommended starting point: alpha=0.5 (KG can contribute up to 50% of max score).
    Sweep range: [0.1, 0.3, 0.5, 1.0, 2.0].

Usage:
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/hybrid_rrf_kg_reranker.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --retrieval_k 20 \\
        --rrf_k 60 \\
        --alpha 0.5 \\
        --output_path evaluation/results/table_retrieval/s1_rrf_kg_reranker.json
"""
import re
from collections import OrderedDict, defaultdict

import config
from logs import logging
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever
from models.retrievers.kg_enriched_bm25_retriever import KGEnrichedBM25Retriever

logger = logging.getLogger(__name__)


class HybridRRFKGReranker(BaseRetriever):
    """B5-a RRF first stage + KG schema-overlap reranker (S1).

    Runs BM25 and ColBERT independently, fuses via RRF (retrieving a pool of
    `retrieval_k` candidates), then re-scores the pool using KG label token
    overlap and returns the top-k.
    """

    def __init__(
        self,
        checkpoint: str,
        collection_variant: str = 'kg_units',
        retrieval_k: int = 20,
        rrf_k: int = 60,
        alpha: float = 0.5,
    ):
        """
        :param checkpoint: ColBERT checkpoint (HuggingFace repo ID or local path).
        :param collection_variant: ColBERT collection variant (e.g. 'kg_units').
        :param retrieval_k: number of candidates fetched from each retriever before fusion.
        :param rrf_k: RRF smoothing constant (paper default: 60).
        :param alpha: KG overlap weight in normalized score space. Scores are
                      normalized to [0,1] (max=1) before adding alpha*overlap.
                      alpha=0.5 → KG can contribute up to 50% of max pool score.
        """
        super().__init__()
        self.retrieval_k = int(retrieval_k)
        self.rrf_k = int(rrf_k)
        self.alpha = float(alpha)

        self.colbert = ColBERTRetriever(
            checkpoint=checkpoint,
            mode='table',
            collection_variant=collection_variant,
        )
        self.bm25 = KGEnrichedBM25Retriever(enrich='True')

        if not config.IS_UNIT_TESTING:
            self.colbert._load_label_tokens()

        if self.colbert.table_label_tokens:
            logger.info(
                f"KG reranker ready: {len(self.colbert.table_label_tokens)} tables, "
                f"alpha={self.alpha}"
            )
        else:
            logger.warning("KG reranker: label tokens not loaded — overlap will be zero.")

    def retrieve_tables(self, question: str, k: int = 10) -> OrderedDict:
        """Retrieve top-k: RRF fusion of BM25+ColBERT, then KG overlap reranking.

        :param question: natural-language question
        :param k: number of tables to return
        :return: OrderedDict {table_id: {'score': combined_score}}
        """
        pool = max(k, self.retrieval_k)

        colbert_results = self.colbert.retrieve_tables(question, k=pool)
        bm25_results = self.bm25.retrieve_tables(question, k=pool)

        rrf_scores: dict[str, float] = defaultdict(float)
        for rank, table_id in enumerate(colbert_results):
            rrf_scores[table_id] += 1.0 / (self.rrf_k + rank + 1)
        for rank, table_id in enumerate(bm25_results):
            rrf_scores[table_id] += 1.0 / (self.rrf_k + rank + 1)

        rrf_pool = OrderedDict(
            (tid, {'score': rrf_scores[tid]})
            for tid in sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        )

        if self.colbert.table_label_tokens:
            reranked = self._kg_overlap_normalized(
                question, rrf_pool, self.colbert.table_label_tokens, self.alpha
            )
            return OrderedDict(list(reranked.items())[:k])

        return OrderedDict(list(rrf_pool.items())[:k])

    @staticmethod
    def _kg_overlap_normalized(
        query: str, pool: OrderedDict, table_label_tokens: dict, alpha: float
    ) -> OrderedDict:
        """Re-score pool with KG overlap using normalized base scores.

        Normalizes each table's RRF score to [0,1] (relative to max in pool)
        before adding alpha * overlap. This makes alpha scale-invariant across
        different first-stage score ranges (ColBERT vs RRF).
        """
        query_tokens = set(re.findall(r'\w+', query.lower()))
        scores_list = [v.get('score', 0) for v in pool.values()]
        max_base = max(scores_list) if scores_list else 1.0

        reranked = {}
        for table_id, data in pool.items():
            label_tokens = table_label_tokens.get(table_id, set())
            overlap = len(query_tokens & label_tokens) / max(len(query_tokens), 1)
            normalized_base = data.get('score', 0) / max_base
            reranked[table_id] = dict(data) | {'combined_score': normalized_base + alpha * overlap}

        return OrderedDict(
            sorted(reranked.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        )

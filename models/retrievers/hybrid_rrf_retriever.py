"""
Hybrid RRF Retriever — BM25 + ColBERT combined via Reciprocal Rank Fusion.

Literature basis:
    Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods." SIGIR 2009.
    Also used as the standard sparse+dense combination in BEIR, DRAGON+, BGE-M3.

Rationale:
    BM25 and ColBERT have complementary failure modes. BM25 fails when the
    question vocabulary differs from the table metadata (vocabulary gap). ColBERT
    fails when semantic similarity flattens across near-duplicate tables. RRF
    combines both ranked lists without requiring score calibration.

Formula (per paper):
    RRF_score(d) = Σ_i  1 / (rrf_k + rank_i(d))
    where rank_i(d) is the 1-based rank of document d in retrieval system i.

Usage (Snellius):
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/hybrid_rrf_retriever.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --retrieval_k 20 \\
        --rrf_k 60 \\
        --output_path evaluation/results/table_retrieval/b5_rrf_k20_rrf60.json

Experiment matrix (B5-a through B5-c):
    B5-a: retrieval_k=20, rrf_k=60  (standard per paper)
    B5-b: retrieval_k=30, rrf_k=60  (wider pool)
    B5-c: retrieval_k=20, rrf_k=10  (low-k RRF variant)
"""
from collections import OrderedDict, defaultdict

from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever
from models.retrievers.kg_enriched_bm25_retriever import KGEnrichedBM25Retriever


class HybridRRFRetriever(BaseRetriever):
    """Two-system retriever: BM25 + ColBERT fused with Reciprocal Rank Fusion.

    Both retrievers are queried with `retrieval_k` candidates each. Their
    ranked lists are merged by RRF and the top-k by fused score are returned.
    """

    def __init__(
        self,
        checkpoint: str,
        collection_variant: str = 'kg_units',
        retrieval_k: int = 20,
        rrf_k: int = 60,
    ):
        """
        :param checkpoint: ColBERT checkpoint (HuggingFace repo ID or local path).
        :param collection_variant: ColBERT collection variant (e.g. 'kg_units').
        :param retrieval_k: number of candidates fetched from each retriever before fusion.
        :param rrf_k: RRF smoothing constant (paper default: 60).
        """
        super().__init__()
        self.retrieval_k = int(retrieval_k)
        self.rrf_k = int(rrf_k)

        self.colbert = ColBERTRetriever(
            checkpoint=checkpoint,
            mode='table',
            collection_variant=collection_variant,
        )
        self.bm25 = KGEnrichedBM25Retriever(enrich='True')

    def retrieve_tables(self, question: str, k: int = 10) -> OrderedDict:
        """Retrieve top-k tables by RRF fusion of BM25 and ColBERT rankings.

        :param question: natural-language question
        :param k: number of tables to return
        :return: OrderedDict {table_id: {'score': rrf_score}}
        """
        pool = max(k, self.retrieval_k)

        colbert_results = self.colbert.retrieve_tables(question, k=pool)
        bm25_results    = self.bm25.retrieve_tables(question, k=pool)

        scores: dict[str, float] = defaultdict(float)
        for rank, table_id in enumerate(colbert_results):
            scores[table_id] += 1.0 / (self.rrf_k + rank + 1)
        for rank, table_id in enumerate(bm25_results):
            scores[table_id] += 1.0 / (self.rrf_k + rank + 1)

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
        return OrderedDict(
            (tid, {'score': scores[tid]}) for tid in sorted_ids[:k]
        )

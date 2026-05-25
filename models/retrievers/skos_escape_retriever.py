"""
SKOS Escape-Pool Retriever — ColBERT + SKOS concept-anchored injection.

Literature basis:
    Liu et al. (2021). "ReTraCk: A Flexible and Efficient Framework for
    Knowledge Base Question Answering." ACL-IJCNLP 2021.

    Gu et al. (2021). "Beyond I.I.D.: Three Levels of Generalization for
    Question Answering on Knowledge Bases." WWW 2021.  (KGPT)

    Shi et al. (2023). "SURGE: Large Language Models and Knowledge Graphs for
    Efficient Retrieval Augmented Generation." (KB-augmented dense retrieval)

Rationale:
    The KG schema-overlap reranker (already in ColBERTRetriever) computes SKOS
    token overlap but only over the top-k pool that ColBERT already returned.
    If the gold table has a unique SKOS dimension/measure label that matches the
    question but ColBERT ranked it at position 25, no reranker can rescue it.

    This retriever extends the mechanism: it builds an inverted index at load
    time (token → set of table IDs whose SKOS labels contain that token) and
    at retrieval time injects any high-confidence matches that are *not* already
    in the ColBERT pool. The augmented pool is then returned for downstream
    reranking (e.g. LLMTableReranker).

    A minimum match threshold of 2 tokens prevents the injection of loosely
    matching tables, preserving precision in the pool.

Usage (Snellius):
    env PYTHONPATH=. LANGUAGE=en python3 evaluation/evaluate_table_retrieval.py \\
        --model_path models/retrievers/skos_escape_retriever.py \\
        --query_type sql --k 10 \\
        --checkpoint StatisticsNetherlands/GECKOv2-ColBERT-EN \\
        --collection_variant kg_units \\
        --colbert_k 20 \\
        --min_token_match 2 \\
        --output_path evaluation/results/table_retrieval/b7_skos_escape_min2.json

Experiment matrix (B7-a through B7-b):
    B7-a: min_token_match=2  (precision-filtered — default)
    B7-b: min_token_match=1  (loose escape, higher recall, lower precision)

Stacking with LLM reranker:
    Replace ColBERTRetriever with SKOSEscapeRetriever inside LLMTableReranker
    by setting --model_path to llm_reranker.py and passing --inner_retriever
    skos_escape to a patched LLMTableReranker, OR run this standalone and pipe
    its output through a separate reranking step.
"""
import json
import os
import re
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, Set

import config
from logs import logging
from models.retrievers.base_retriever import BaseRetriever
from models.retrievers.colbert.colbert_retriever import ColBERTRetriever

logger = logging.getLogger(__name__)

BASE_PATH = f"{config.PATH_DIR_DATA}/colbert_retriever"
MIN_TOKEN_LEN = 4  # skip short stop-words in the inverted index

# Generic statistical tokens that appear in hundreds of tables — indexing them
# would cause noise injections and are excluded from the inverted index.
_STOPWORDS = {
    "total", "count", "index", "value", "amount", "share", "change",
    "number", "average", "annual", "monthly", "weekly", "quarterly",
    "other", "general", "male", "female", "both", "year", "period",
    "margin", "percent", "rate", "mean", "from", "with", "that", "this",
    "have", "been", "were", "they", "their",
}


def _build_skos_inverted(table_label_tokens: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Build token → set(table_id) inverted index from per-table SKOS token sets.

    Filters out tokens shorter than MIN_TOKEN_LEN and tokens in _STOPWORDS to
    avoid high-frequency spurious matches.
    """
    inv: Dict[str, Set[str]] = defaultdict(set)
    for table_id, token_set in table_label_tokens.items():
        for tok in token_set:
            if len(tok) >= MIN_TOKEN_LEN and tok not in _STOPWORDS:
                inv[tok].add(table_id)
    return dict(inv)


class SKOSEscapeRetriever(BaseRetriever):
    """ColBERT retriever augmented with SKOS-concept-anchored candidate injection.

    The retriever runs ColBERT first (recall stage), then scans all 2,244 tables
    for SKOS label matches against the question. Tables with ≥ min_token_match
    matching tokens that are *not already in the ColBERT pool* are appended to
    the candidate set before the final top-k slice.

    Because injected tables are appended after the ColBERT pool with a score
    below any ColBERT candidate, the ranking within the ColBERT pool is
    preserved. Downstream rerankers (KG overlap, LLM) can re-order the full
    augmented pool.
    """

    def __init__(
        self,
        checkpoint: str,
        collection_variant: str = 'kg_units',
        colbert_k: int = 20,
        min_token_match: int = 2,
    ):
        """
        :param checkpoint: ColBERT checkpoint (HuggingFace repo ID or local path).
        :param collection_variant: ColBERT collection variant (e.g. 'kg_units').
        :param colbert_k: number of candidates fetched from ColBERT before injection.
        :param min_token_match: minimum number of SKOS tokens that must match the
                                question for a table to be injected into the pool.
                                Higher = more precise, fewer injections.
                                Lower = more injections, risk of noise.
        """
        super().__init__()
        self.colbert_k = int(colbert_k)
        self.min_token_match = int(min_token_match)

        self.colbert = ColBERTRetriever(
            checkpoint=checkpoint,
            mode='table',
            collection_variant=collection_variant,
        )
        # Load the SKOS label token cache independently of the kg_rerank flag.
        # We need the tokens for the inverted index but don't want the ColBERT
        # inner call to waste time running the overlap reranker on every query.
        if not config.IS_UNIT_TESTING:
            self.colbert._load_label_tokens()

        if self.colbert.table_label_tokens:
            self._inverted = _build_skos_inverted(self.colbert.table_label_tokens)
            logger.info(
                f"SKOS inverted index: {len(self._inverted)} unique tokens "
                f"over {len(self.colbert.table_label_tokens)} tables"
            )
        else:
            self._inverted = {}
            logger.warning(
                "SKOS inverted index is empty — node_labels cache not loaded. "
                "Run colbert_trainer.py --rebuild_collection first."
            )

    def _skos_escape_candidates(
        self, question: str, already_in_pool: set
    ) -> list[tuple[str, int]]:
        """Find tables with ≥ min_token_match SKOS label tokens matching the question
        that are not already in the ColBERT pool.

        Returns a list of (table_id, match_count) sorted descending by match count.
        """
        q_tokens = set(re.findall(r'\b[a-z]{4,}\b', question.lower()))
        candidate_scores: Counter = Counter()
        for tok in q_tokens:
            for table_id in self._inverted.get(tok, set()):
                if table_id not in already_in_pool:
                    candidate_scores[table_id] += 1

        return [
            (tid, cnt)
            for tid, cnt in candidate_scores.most_common()
            if cnt >= self.min_token_match
        ]

    def retrieve_tables(self, question: str, k: int = 10) -> OrderedDict:
        """Retrieve top-k tables: ColBERT pool + SKOS-escaped candidates.

        ColBERT candidates retain their original scores. Injected tables are
        assigned a score slightly below the lowest ColBERT score, ordered by
        SKOS match count.

        :param question: natural-language question
        :param k: number of tables to return
        :return: OrderedDict {table_id: {'score': float}}
        """
        pool_k = max(k, self.colbert_k)
        colbert_results = self.colbert.retrieve_tables(question, k=pool_k)
        already_in_pool = set(colbert_results.keys())

        escaped = self._skos_escape_candidates(question, already_in_pool)

        if not escaped:
            return OrderedDict(list(colbert_results.items())[:k])

        # Score floor: just below the lowest ColBERT score so injection doesn't
        # displace ColBERT candidates within the requested top-k.
        colbert_scores = [v.get('score', 0.0) for v in colbert_results.values()]
        floor = min(colbert_scores) * 0.99 if colbert_scores else 0.0

        result = OrderedDict(colbert_results)
        for i, (tid, match_count) in enumerate(escaped):
            result[tid] = {'score': floor - i * 1e-6, 'skos_match_count': match_count}

        if escaped:
            logger.debug(
                f"SKOS escape: injected {len(escaped)} table(s) for '{question[:60]}...'"
            )

        return OrderedDict(list(result.items())[:k])

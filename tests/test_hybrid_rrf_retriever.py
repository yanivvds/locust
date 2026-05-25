"""
Unit tests for HybridRRFRetriever.

Tests the RRF fusion logic in isolation — no ColBERT model, no BM25 index,
no graph required. The two inner retrievers are replaced with mock objects
that return controlled ranked lists.
"""
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import config
config.IS_UNIT_TESTING = True

from models.retrievers.hybrid_rrf_retriever import HybridRRFRetriever


def _make_retriever(colbert_results: dict, bm25_results: dict) -> HybridRRFRetriever:
    """Build a HybridRRFRetriever with mocked inner retrievers."""
    with (
        patch("models.retrievers.hybrid_rrf_retriever.ColBERTRetriever"),
        patch("models.retrievers.hybrid_rrf_retriever.KGEnrichedBM25Retriever"),
    ):
        r = HybridRRFRetriever.__new__(HybridRRFRetriever)
        r.retrieval_k = 10
        r.rrf_k = 60

        r.colbert = MagicMock()
        r.colbert.retrieve_tables.return_value = OrderedDict(
            (tid, {"score": s}) for tid, s in colbert_results.items()
        )

        r.bm25 = MagicMock()
        r.bm25.retrieve_tables.return_value = OrderedDict(
            (tid, {"score": s}) for tid, s in bm25_results.items()
        )
        return r


class TestRRFScoring:
    def test_table_in_both_lists_gets_double_score(self):
        """A table ranked #1 by both retrievers must outscore a table ranked #1 by only one."""
        r = _make_retriever(
            colbert_results={"t_both": 10.0, "t_only_colbert": 9.0},
            bm25_results={"t_both": 5.0, "t_only_bm25": 4.0},
        )
        result = r.retrieve_tables("any question", k=3)
        ids = list(result.keys())
        assert ids[0] == "t_both", "table present in both lists should be ranked first"

    def test_rrf_formula_values(self):
        """Verify the exact RRF score for a known rank assignment."""
        r = _make_retriever(
            colbert_results={"t1": 10.0},   # rank 1 in colbert
            bm25_results={"t1": 5.0},       # rank 1 in bm25
        )
        result = r.retrieve_tables("any question", k=1)
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)  # 2 * 1/61
        assert abs(result["t1"]["score"] - expected) < 1e-9

    def test_colbert_only_table_included(self):
        """A table retrieved by only ColBERT (not BM25) is still included in output."""
        r = _make_retriever(
            colbert_results={"t_colbert": 10.0},
            bm25_results={"t_bm25": 5.0},
        )
        result = r.retrieve_tables("any question", k=2)
        assert "t_colbert" in result
        assert "t_bm25" in result

    def test_bm25_only_table_included(self):
        """A table retrieved by only BM25 (not ColBERT) is still included in output."""
        r = _make_retriever(
            colbert_results={"t_colbert": 10.0},
            bm25_results={"t_bm25_only": 5.0},
        )
        result = r.retrieve_tables("any question", k=2)
        assert "t_bm25_only" in result

    def test_top_k_respected(self):
        """Output is capped at k tables even when the pool is larger."""
        r = _make_retriever(
            colbert_results={f"c{i}": float(10 - i) for i in range(8)},
            bm25_results={f"b{i}": float(8 - i) for i in range(8)},
        )
        result = r.retrieve_tables("any question", k=5)
        assert len(result) == 5

    def test_result_is_ordered_dict(self):
        """Output type must be OrderedDict (matches BaseRetriever interface)."""
        r = _make_retriever({"t1": 5.0}, {"t2": 3.0})
        result = r.retrieve_tables("question", k=2)
        assert isinstance(result, OrderedDict)

    def test_descending_score_order(self):
        """Tables must be ordered from highest to lowest RRF score."""
        r = _make_retriever(
            colbert_results={"t1": 1.0, "t2": 0.9, "t3": 0.8},
            bm25_results={"t3": 1.0, "t2": 0.9, "t1": 0.8},
        )
        result = r.retrieve_tables("question", k=3)
        scores = [v["score"] for v in result.values()]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_k_parameter_affects_scores(self):
        """A larger rrf_k constant should produce lower absolute scores."""
        r_small = _make_retriever({"t1": 1.0}, {"t1": 1.0})
        r_large = _make_retriever({"t1": 1.0}, {"t1": 1.0})
        r_small.rrf_k = 10
        r_large.rrf_k = 100

        score_small = r_small.retrieve_tables("q", k=1)["t1"]["score"]
        score_large = r_large.retrieve_tables("q", k=1)["t1"]["score"]
        assert score_small > score_large

    def test_rank_based_not_score_based(self):
        """A low-scored ColBERT table ranked #1 should still beat a high-scored #2 table.

        RRF uses rank position, not raw retriever scores — so ColBERT scores
        in the input dicts should not affect the fusion outcome.
        """
        r = _make_retriever(
            colbert_results={"t_rank1": 0.01, "t_rank2": 99.0},  # t_rank1 is rank 1
            bm25_results={},
        )
        result = r.retrieve_tables("question", k=2)
        ids = list(result.keys())
        assert ids[0] == "t_rank1"

    def test_empty_bm25_results(self):
        """Works correctly when BM25 returns nothing (e.g. empty corpus)."""
        r = _make_retriever(
            colbert_results={"t1": 1.0, "t2": 0.9},
            bm25_results={},
        )
        result = r.retrieve_tables("question", k=2)
        assert list(result.keys()) == ["t1", "t2"]

    def test_empty_colbert_results(self):
        """Works correctly when ColBERT returns nothing."""
        r = _make_retriever(
            colbert_results={},
            bm25_results={"t1": 1.0},
        )
        result = r.retrieve_tables("question", k=1)
        assert "t1" in result

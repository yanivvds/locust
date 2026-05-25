"""
Unit tests for SKOSEscapeRetriever.

Tests the SKOS inverted index construction and candidate injection logic in
isolation — no ColBERT model, no graph, no node_labels JSON required.
The inner ColBERT retriever is mocked; table_label_tokens is set directly.
"""
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import config
config.IS_UNIT_TESTING = True

from models.retrievers.skos_escape_retriever import (
    SKOSEscapeRetriever,
    _build_skos_inverted,
    _STOPWORDS,
    MIN_TOKEN_LEN,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_retriever(
    colbert_results: dict,
    table_label_tokens: dict,
    min_token_match: int = 2,
    colbert_k: int = 5,
) -> SKOSEscapeRetriever:
    """Build an SKOSEscapeRetriever with mocked ColBERT and controlled tokens."""
    with patch("models.retrievers.skos_escape_retriever.ColBERTRetriever"):
        r = SKOSEscapeRetriever.__new__(SKOSEscapeRetriever)
        r.colbert_k = colbert_k
        r.min_token_match = min_token_match

        r.colbert = MagicMock()
        r.colbert.retrieve_tables.return_value = OrderedDict(
            (tid, {"score": s}) for tid, s in colbert_results.items()
        )
        r.colbert.table_label_tokens = table_label_tokens

        r._inverted = _build_skos_inverted(table_label_tokens)
        return r


# ──────────────────────────────────────────────
# _build_skos_inverted
# ──────────────────────────────────────────────

class TestBuildSkosInverted:
    def test_basic_mapping(self):
        tokens = {"t1": {"concentrated", "milk"}, "t2": {"nitrogen", "emissions"}}
        inv = _build_skos_inverted(tokens)
        assert "t1" in inv["concentrated"]
        assert "t1" in inv["milk"]
        assert "t2" in inv["nitrogen"]

    def test_token_shared_across_tables(self):
        tokens = {"t1": {"employment"}, "t2": {"employment", "sector"}}
        inv = _build_skos_inverted(tokens)
        assert "t1" in inv["employment"]
        assert "t2" in inv["employment"]

    def test_short_tokens_excluded(self):
        tokens = {"t1": {"of", "the", "in"}}
        inv = _build_skos_inverted(tokens)
        for short_tok in ("of", "the", "in"):
            assert short_tok not in inv

    def test_stopwords_excluded(self):
        for word in _STOPWORDS:
            tokens = {"t1": {word}}
            inv = _build_skos_inverted(tokens)
            assert word not in inv, f"stopword '{word}' should be excluded"

    def test_min_token_len_boundary(self):
        """Token of exactly MIN_TOKEN_LEN characters should be included."""
        tok = "a" * MIN_TOKEN_LEN
        tokens = {"t1": {tok}}
        inv = _build_skos_inverted(tokens)
        assert tok in inv

    def test_empty_input(self):
        assert _build_skos_inverted({}) == {}

    def test_all_filtered_tokens(self):
        """If every token is short or a stopword, result is empty."""
        tokens = {"t1": {"of", "the", "total", "rate"}}  # all filtered
        inv = _build_skos_inverted(tokens)
        assert len(inv) == 0


# ──────────────────────────────────────────────
# _skos_escape_candidates
# ──────────────────────────────────────────────

class TestSkosEscapeCandidates:
    def test_injects_missing_table(self):
        """A table whose SKOS label matches the question but is absent from the pool
        should appear in escape candidates."""
        r = _make_retriever(
            colbert_results={"t_in_pool": 1.0},
            table_label_tokens={
                "t_in_pool":    {"electricity", "production"},
                "t_escaped":    {"electricity", "production", "renewable"},
            },
            min_token_match=2,
        )
        escaped = r._skos_escape_candidates(
            "What is the electricity production from renewable sources?",
            already_in_pool={"t_in_pool"},
        )
        escaped_ids = [tid for tid, _ in escaped]
        assert "t_escaped" in escaped_ids

    def test_does_not_inject_pool_tables(self):
        """Tables already in the ColBERT pool must not be injected again."""
        r = _make_retriever(
            colbert_results={"t1": 1.0},
            table_label_tokens={"t1": {"electricity", "production"}},
            min_token_match=1,
        )
        escaped = r._skos_escape_candidates(
            "electricity production",
            already_in_pool={"t1"},
        )
        escaped_ids = [tid for tid, _ in escaped]
        assert "t1" not in escaped_ids

    def test_min_token_match_filters_weak_matches(self):
        """Table matching only 1 token is excluded when min_token_match=2."""
        r = _make_retriever(
            colbert_results={},
            table_label_tokens={"t_weak": {"electricity"}},
            min_token_match=2,
        )
        escaped = r._skos_escape_candidates("electricity output", already_in_pool=set())
        escaped_ids = [tid for tid, _ in escaped]
        assert "t_weak" not in escaped_ids

    def test_min_token_match_one_allows_weak_match(self):
        """With min_token_match=1, a single matching token is enough to inject."""
        r = _make_retriever(
            colbert_results={},
            table_label_tokens={"t_single": {"electricity"}},
            min_token_match=1,
        )
        escaped = r._skos_escape_candidates("electricity output", already_in_pool=set())
        escaped_ids = [tid for tid, _ in escaped]
        assert "t_single" in escaped_ids

    def test_sorted_by_match_count_descending(self):
        """Candidates with more matching tokens should come first."""
        r = _make_retriever(
            colbert_results={},
            table_label_tokens={
                "t_3matches": {"renewable", "electricity", "production"},
                "t_2matches": {"renewable", "electricity"},
            },
            min_token_match=2,
        )
        question = "What is the renewable electricity production?"
        escaped = r._skos_escape_candidates(question, already_in_pool=set())
        assert escaped[0][0] == "t_3matches"

    def test_empty_inverted_index_returns_empty(self):
        r = _make_retriever(
            colbert_results={},
            table_label_tokens={},
            min_token_match=1,
        )
        escaped = r._skos_escape_candidates("anything", already_in_pool=set())
        assert escaped == []


# ──────────────────────────────────────────────
# retrieve_tables (integration of the two stages)
# ──────────────────────────────────────────────

class TestRetrieveTables:
    def test_colbert_results_preserved_at_top(self):
        """ColBERT candidates must appear before injected SKOS candidates."""
        r = _make_retriever(
            colbert_results={"colbert_1": 10.0, "colbert_2": 9.0},
            table_label_tokens={
                "colbert_1": set(),
                "colbert_2": set(),
                "escaped_1": {"renewable", "electricity"},
            },
            min_token_match=2,
        )
        result = r.retrieve_tables(
            "What is the renewable electricity production?", k=3
        )
        ids = list(result.keys())
        assert ids[0] == "colbert_1"
        assert ids[1] == "colbert_2"
        assert ids[2] == "escaped_1"

    def test_injected_score_below_colbert_floor(self):
        """Injected table score must be strictly below all ColBERT scores."""
        r = _make_retriever(
            colbert_results={"colbert_1": 5.0},
            table_label_tokens={"escaped_1": {"renewable", "electricity"}},
            min_token_match=2,
        )
        result = r.retrieve_tables("renewable electricity generation", k=2)
        assert result["escaped_1"]["score"] < result["colbert_1"]["score"]

    def test_k_limits_output(self):
        """Output is capped at k even when ColBERT + escaped pool is larger."""
        r = _make_retriever(
            colbert_results={f"c{i}": float(5 - i) for i in range(4)},
            table_label_tokens={
                "escaped_1": {"unique", "mineral", "fertilizer"},
                "escaped_2": {"unique", "mineral", "potassium"},
            },
            min_token_match=2,
        )
        result = r.retrieve_tables("unique mineral use", k=3)
        assert len(result) == 3

    def test_no_injection_when_no_matches(self):
        """If no SKOS labels match the question, the output equals plain ColBERT top-k."""
        r = _make_retriever(
            colbert_results={"c1": 1.0, "c2": 0.9},
            table_label_tokens={"other": {"xyzzyx", "noquestion"}},
            min_token_match=2,
        )
        result = r.retrieve_tables("completely different query", k=2)
        assert list(result.keys()) == ["c1", "c2"]

    def test_result_is_ordered_dict(self):
        r = _make_retriever({"c1": 1.0}, {"c1": {"word"}}, min_token_match=1)
        result = r.retrieve_tables("word query", k=1)
        assert isinstance(result, OrderedDict)

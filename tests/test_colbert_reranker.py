"""
Unit tests for ColBERTRetriever KG overlap reranker static methods.

Tests _build_table_label_tokens and _compute_kg_overlap in isolation —
no ColBERT model, no index, no graph required.
"""
from collections import OrderedDict

import config
config.IS_UNIT_TESTING = True  # prevents index_colbert() from running in __init__

from models.retrievers.colbert.colbert_retriever import ColBERTRetriever


class TestBuildTableLabelTokens:
    def test_skips_table_type_entries(self):
        """Table-level entries (type='table') must not contribute tokens to the token set."""
        node_labels = {
            "81156eng": {"body": "Trade and industry.", "type": "table"},
            "81156eng#m1": {
                "body": "ignored body",
                "type": "measure",
                "table": "81156eng",
                "prefLabel": "Employment",
            },
        }
        result = ColBERTRetriever._build_table_label_tokens(node_labels)
        assert "81156eng" in result
        assert "employment" in result["81156eng"]
        # Table body text should NOT appear
        assert "trade" not in result["81156eng"]
        assert "industry" not in result["81156eng"]

    def test_uses_prefLabel_when_present(self):
        """prefLabel takes priority over body when non-empty."""
        node_labels = {
            "t1#m1": {
                "body": "body words only",
                "type": "measure",
                "table": "t1",
                "prefLabel": "Preferred Label Text",
            }
        }
        result = ColBERTRetriever._build_table_label_tokens(node_labels)
        assert "preferred" in result["t1"]
        assert "label" in result["t1"]
        assert "text" in result["t1"]
        assert "body" not in result["t1"]
        assert "words" not in result["t1"]

    def test_falls_back_to_body_when_prefLabel_empty(self):
        """Body text is used when prefLabel is absent or an empty string."""
        node_labels = {
            "t1#m1": {
                "body": "Profit and loss.",
                "type": "measure",
                "table": "t1",
                "prefLabel": "",
            },
        }
        result = ColBERTRetriever._build_table_label_tokens(node_labels)
        assert "profit" in result["t1"]
        assert "loss" in result["t1"]

    def test_falls_back_to_body_when_prefLabel_absent(self):
        """Body text is used when the 'prefLabel' key is missing entirely."""
        node_labels = {
            "t1#d1": {"body": "Revenue growth.", "type": "dimension", "table": "t1"},
        }
        result = ColBERTRetriever._build_table_label_tokens(node_labels)
        assert "revenue" in result["t1"]
        assert "growth" in result["t1"]

    def test_aggregates_multiple_nodes_per_table(self):
        """All nodes belonging to the same table are merged into one token set."""
        node_labels = {
            "t1#m1": {"body": "", "type": "measure", "table": "t1", "prefLabel": "Revenue"},
            "t1#d1": {"body": "", "type": "dimension", "table": "t1", "prefLabel": "Sector"},
        }
        result = ColBERTRetriever._build_table_label_tokens(node_labels)
        assert "revenue" in result["t1"]
        assert "sector" in result["t1"]

    def test_missing_table_field_falls_back_to_key_split(self):
        """If 'table' field is absent, table_id is derived from the '#'-split key prefix."""
        node_labels = {
            "t2#m1": {"body": "", "type": "measure", "prefLabel": "Wages"},
        }
        result = ColBERTRetriever._build_table_label_tokens(node_labels)
        assert "t2" in result
        assert "wages" in result["t2"]

    def test_empty_node_labels_returns_empty_dict(self):
        result = ColBERTRetriever._build_table_label_tokens({})
        assert result == {}

    def test_all_table_types_returns_empty_dict(self):
        """If every entry is type='table', result has no token sets."""
        node_labels = {
            "t1": {"body": "Some table description.", "type": "table"},
        }
        result = ColBERTRetriever._build_table_label_tokens(node_labels)
        assert result == {}


class TestComputeKgOverlap:
    def _make_ranked(self, items: list) -> OrderedDict:
        return OrderedDict((tid, {"score": score}) for tid, score in items)

    def test_higher_overlap_wins_when_alpha_is_large(self):
        """Table with more query-token matches should rank first when alpha dominates."""
        ranked = self._make_ranked([("t1", 10.0), ("t2", 9.5)])
        label_tokens = {
            "t1": {"employees", "sector"},
            "t2": {"employment", "manufacturing", "sector", "industry"},
        }
        # Query has 5 tokens; t1 matches 1 ('sector'), t2 matches 3 ('employment','manufacturing','industry')
        result = ColBERTRetriever._compute_kg_overlap(
            "How many employment in manufacturing industry?",
            ranked, label_tokens, alpha=10.0
        )
        assert list(result.keys())[0] == "t2"

    def test_alpha_zero_preserves_original_order(self):
        """When alpha=0, no overlap bonus is added and original ColBERT score order holds."""
        ranked = self._make_ranked([("t1", 10.0), ("t2", 9.5)])
        label_tokens = {"t1": set(), "t2": {"query", "words", "here"}}
        result = ColBERTRetriever._compute_kg_overlap(
            "query words here", ranked, label_tokens, alpha=0.0
        )
        assert list(result.keys())[0] == "t1"

    def test_missing_table_in_label_tokens_scores_zero_overlap(self):
        """Tables absent from label_tokens get overlap=0 — no KeyError."""
        ranked = self._make_ranked([("t1", 10.0), ("t_unknown", 11.0)])
        label_tokens = {"t1": {"revenue"}}
        # t_unknown has no entry => overlap 0; score 11.0
        # t1 overlap = 1/2 = 0.5; combined = 10.0 + 1.0*0.5 = 10.5 < 11.0
        result = ColBERTRetriever._compute_kg_overlap(
            "revenue growth", ranked, label_tokens, alpha=1.0
        )
        assert list(result.keys())[0] == "t_unknown"

    def test_combined_score_stored_in_result(self):
        """Every entry in the returned dict must carry a 'combined_score' key."""
        ranked = self._make_ranked([("t1", 5.0)])
        result = ColBERTRetriever._compute_kg_overlap(
            "test query", ranked, {"t1": {"test"}}, alpha=1.0
        )
        assert "combined_score" in result["t1"]

    def test_existing_combined_score_used_as_base(self):
        """In 'all' mode, ranked_items already has combined_score — reranker adds on top of it."""
        ranked = OrderedDict([("t1", {"score": 5.0, "combined_score": 8.0})])
        label_tokens = {"t1": {"foo"}}
        # Query "foo bar" has 2 tokens; t1 matches 1 => overlap = 0.5
        # combined = 8.0 + 2.0 * 0.5 = 9.0
        result = ColBERTRetriever._compute_kg_overlap(
            "foo bar", ranked, label_tokens, alpha=2.0
        )
        assert abs(result["t1"]["combined_score"] - 9.0) < 1e-6

    def test_empty_query_gives_zero_overlap(self):
        """Empty query string produces no tokens; overlap is 0 for all tables."""
        ranked = self._make_ranked([("t1", 5.0), ("t2", 4.0)])
        label_tokens = {"t1": {"anything"}, "t2": {"something"}}
        result = ColBERTRetriever._compute_kg_overlap("", ranked, label_tokens, alpha=10.0)
        # With overlap=0 for all, original score order must be preserved
        assert list(result.keys())[0] == "t1"

    def test_original_data_not_mutated(self):
        """ranked_items dict entries must not be modified in-place."""
        ranked = self._make_ranked([("t1", 5.0)])
        original_data = dict(ranked["t1"])
        ColBERTRetriever._compute_kg_overlap("test", ranked, {"t1": {"test"}}, alpha=1.0)
        assert ranked["t1"] == original_data

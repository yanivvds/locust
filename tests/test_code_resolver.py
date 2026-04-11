"""
Unit tests for SKOSCodeResolver and the get_dimension_codes() SPARQL helper.

Uses the unit test graph (ut_graph.trig) with tables 80781ned, 84957NED, 85302NED.
No LLM or external API calls are made.
"""
import pytest

from models.generators.code_resolver import SKOSCodeResolver
from s_expression import Table


# --- get_dimension_codes() SPARQL helper ---

class TestGetDimensionCodes:
    def test_returns_nonempty_for_known_dim(self, sparql_engine):
        """85302NED has a BestemmingEnSeizoen dimension with multiple codes."""
        table = Table('85302NED')
        # Get schema dim IDs via the resolver helper, pick the first one
        resolver = SKOSCodeResolver(sparql_engine)
        dim_ids = resolver._get_schema_dim_ids(table)
        assert len(dim_ids) > 0, "Expected at least one schema dimension on 85302NED"

        codes = sparql_engine.get_dimension_codes(table, dim_ids[0])
        assert isinstance(codes, dict)
        assert len(codes) > 0, f"Expected codes for dim {dim_ids[0]} on 85302NED"

    def test_returns_empty_for_unknown_dim(self, sparql_engine):
        """A dimension ID that doesn't exist should return an empty dict, not crash."""
        table = Table('85302NED')
        codes = sparql_engine.get_dimension_codes(table, 'NonExistentDim')
        assert codes == {}

    def test_code_label_types_are_strings(self, sparql_engine):
        """Keys and values must both be plain strings."""
        table = Table('85302NED')
        resolver = SKOSCodeResolver(sparql_engine)
        dim_ids = resolver._get_schema_dim_ids(table)
        if not dim_ids:
            pytest.skip("No schema dims found on 85302NED")

        codes = sparql_engine.get_dimension_codes(table, dim_ids[0])
        for code, label in codes.items():
            assert isinstance(code, str), f"Code should be str, got {type(code)}"
            assert isinstance(label, str), f"Label should be str, got {type(label)}"


# --- SKOSCodeResolver ---

class TestSKOSCodeResolver:
    def test_resolve_returns_dict(self, sparql_engine):
        """resolve() always returns a dict, even when nothing matches."""
        resolver = SKOSCodeResolver(sparql_engine)
        result = resolver.resolve("some random question", "85302NED")
        assert isinstance(result, dict)

    def test_resolve_returns_empty_for_unrelated_question(self, sparql_engine):
        """A completely unrelated question should match nothing at high threshold."""
        resolver = SKOSCodeResolver(sparql_engine, threshold=90)
        result = resolver.resolve("zzz xyz this matches nothing at all", "85302NED")
        assert result == {}

    def test_resolve_finds_match_with_zero_threshold(self, sparql_engine):
        """
        At threshold=0, every question gets a match for every dim that has codes.
        This verifies the SPARQL → fuzzy-match pipeline runs end-to-end.
        """
        resolver = SKOSCodeResolver(sparql_engine, threshold=0)
        result = resolver.resolve("any question at all", "85302NED")
        # Should have at least one resolved dim at threshold=0
        assert len(result) > 0, "Expected at least one match at threshold=0"

    def test_resolve_codes_are_valid_strings(self, sparql_engine):
        """All returned codes should be non-empty strings."""
        resolver = SKOSCodeResolver(sparql_engine, threshold=0)
        result = resolver.resolve("any question", "85302NED")
        for dim_id, code in result.items():
            assert isinstance(dim_id, str) and dim_id
            assert isinstance(code, str) and code

    def test_threshold_filters_low_confidence_matches(self, sparql_engine):
        """High threshold (100) should return fewer or equal matches than low threshold (0)."""
        resolver_strict = SKOSCodeResolver(sparql_engine, threshold=100)
        resolver_loose = SKOSCodeResolver(sparql_engine, threshold=0)
        question = "any question"
        strict_result = resolver_strict.resolve(question, "85302NED")
        loose_result = resolver_loose.resolve(question, "85302NED")
        assert len(strict_result) <= len(loose_result)

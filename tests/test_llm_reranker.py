"""Unit tests for LLMTableReranker static methods.

No ColBERT index, no vLLM server, and no graph required — tests only cover
the three static methods that are fully deterministic.
"""
from collections import OrderedDict

import pytest

from models.retrievers.llm_reranker import LLMTableReranker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POOL = ['85055ENG', '84957NED', '80781NED', '85302NED', '70001NED']
VALID_UPPER = {tid.upper(): tid for tid in POOL}


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:

    def test_numbered_prefix_stripped(self):
        raw = "1. 85055ENG\n2. 84957NED"
        result = LLMTableReranker._parse_response(raw, VALID_UPPER, POOL)
        assert result[0] == '85055ENG'
        assert result[1] == '84957NED'

    def test_parenthesis_prefix_stripped(self):
        raw = "1) 85055ENG\n2) 84957NED"
        result = LLMTableReranker._parse_response(raw, VALID_UPPER, POOL)
        assert result[0] == '85055ENG'
        assert result[1] == '84957NED'

    def test_colon_separator_handled(self):
        raw = "85055ENG: Trade in goods\n84957NED: Population"
        result = LLMTableReranker._parse_response(raw, VALID_UPPER, POOL)
        assert result[0] == '85055ENG'
        assert result[1] == '84957NED'

    def test_unknown_ids_dropped(self):
        raw = "UNKNOWN123\n85055ENG\nANOTHERBAD"
        result = LLMTableReranker._parse_response(raw, VALID_UPPER, POOL)
        assert result[0] == '85055ENG'
        assert 'UNKNOWN123' not in result
        assert 'ANOTHERBAD' not in result

    def test_partial_list_padded_with_remaining_colbert_order(self):
        raw = "85055ENG\n84957NED"
        result = LLMTableReranker._parse_response(raw, VALID_UPPER, POOL)
        # First two from LLM, rest in original ColBERT order
        assert result[:2] == ['85055ENG', '84957NED']
        assert set(result) == set(POOL)
        assert result[2:] == ['80781NED', '85302NED', '70001NED']

    def test_empty_response_returns_fallback_unchanged(self):
        result = LLMTableReranker._parse_response("", VALID_UPPER, POOL)
        assert result == POOL

    def test_all_unknown_returns_fallback_unchanged(self):
        raw = "BADID1\nBADID2\nBADID3"
        result = LLMTableReranker._parse_response(raw, VALID_UPPER, POOL)
        assert result == POOL

    def test_deduplication(self):
        raw = "85055ENG\n85055ENG\n84957NED"
        result = LLMTableReranker._parse_response(raw, VALID_UPPER, POOL)
        assert result.count('85055ENG') == 1
        assert result[0] == '85055ENG'

    def test_case_insensitive_match(self):
        raw = "85055eng\n84957ned"
        result = LLMTableReranker._parse_response(raw, VALID_UPPER, POOL)
        # Should resolve to the original-case IDs from the pool
        assert result[0] == '85055ENG'
        assert result[1] == '84957NED'

    def test_result_contains_all_pool_ids(self):
        raw = "85055ENG\n80781NED"
        result = LLMTableReranker._parse_response(raw, VALID_UPPER, POOL)
        assert set(result) == set(POOL)
        assert len(result) == len(POOL)


# ---------------------------------------------------------------------------
# _build_table_info
# ---------------------------------------------------------------------------

class TestBuildTableInfo:

    def test_without_description(self):
        table_ids = ['85055ENG']
        titles = {'85055ENG': {'title': 'Trade in goods', 'description': ''}}
        result = LLMTableReranker._build_table_info(table_ids, titles)
        assert result == '1. 85055ENG: Trade in goods'

    def test_with_description(self):
        table_ids = ['85055ENG']
        titles = {'85055ENG': {'title': 'Trade', 'description': 'Imports and exports by country'}}
        result = LLMTableReranker._build_table_info(table_ids, titles)
        assert '— Imports and exports by country' in result

    def test_description_truncated_at_120_chars(self):
        long_desc = 'x' * 200
        table_ids = ['85055ENG']
        titles = {'85055ENG': {'title': 'Trade', 'description': long_desc}}
        result = LLMTableReranker._build_table_info(table_ids, titles)
        # The description portion should be exactly 120 chars
        desc_part = result.split('— ')[1]
        assert len(desc_part) == 120

    def test_description_omitted_when_identical_to_title(self):
        table_ids = ['85055ENG']
        titles = {'85055ENG': {'title': 'Trade in goods', 'description': 'Trade in goods'}}
        result = LLMTableReranker._build_table_info(table_ids, titles)
        assert '—' not in result

    def test_multiple_tables_numbered_correctly(self):
        table_ids = ['85055ENG', '84957NED']
        titles = {
            '85055ENG': {'title': 'Trade', 'description': ''},
            '84957NED': {'title': 'Population', 'description': ''},
        }
        result = LLMTableReranker._build_table_info(table_ids, titles)
        lines = result.splitlines()
        assert lines[0].startswith('1.')
        assert lines[1].startswith('2.')

    def test_missing_title_falls_back_to_id(self):
        table_ids = ['85055ENG']
        titles = {}
        result = LLMTableReranker._build_table_info(table_ids, titles)
        assert '85055ENG: 85055ENG' in result


# ---------------------------------------------------------------------------
# _build_result
# ---------------------------------------------------------------------------

class TestBuildResult:

    def test_scores_are_reciprocal_rank(self):
        ids = ['A', 'B', 'C']
        result = LLMTableReranker._build_result(ids, k=3)
        assert result['A']['score'] == pytest.approx(1.0)
        assert result['B']['score'] == pytest.approx(0.5)
        assert result['C']['score'] == pytest.approx(1.0 / 3)

    def test_sliced_to_k(self):
        ids = ['A', 'B', 'C', 'D', 'E']
        result = LLMTableReranker._build_result(ids, k=3)
        assert len(result) == 3
        assert 'D' not in result
        assert 'E' not in result

    def test_returns_ordered_dict(self):
        ids = ['A', 'B', 'C']
        result = LLMTableReranker._build_result(ids, k=3)
        assert isinstance(result, OrderedDict)

    def test_order_preserved(self):
        ids = ['C', 'A', 'B']
        result = LLMTableReranker._build_result(ids, k=3)
        assert list(result.keys()) == ['C', 'A', 'B']

    def test_k_larger_than_pool_returns_full_pool(self):
        ids = ['A', 'B']
        result = LLMTableReranker._build_result(ids, k=10)
        assert len(result) == 2

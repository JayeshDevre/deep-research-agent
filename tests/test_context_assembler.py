"""
Tests for context assembler: chunk packing, relevance sorting, token budget.
Run with: pytest tests/
"""

import pytest
from agent.context_assembler import assemble
from agent.budget import count_tokens
from agent.config import CONTEXT_TOKEN_LIMIT


def make_chunk(text: str, score: float = 0.5, source: str = "test") -> dict:
    return {"text": text, "score": score, "source": source}


class TestAssembleBasic:
    def test_empty_input_returns_empty_string(self):
        assert assemble([]) == ""

    def test_single_chunk_included(self):
        chunk  = make_chunk("some research content")
        result = assemble([chunk], reserve_tokens=0)
        assert "some research content" in result

    def test_source_label_included(self):
        chunk  = make_chunk("content", source="https://example.com")
        result = assemble([chunk], reserve_tokens=0)
        assert "https://example.com" in result

    def test_missing_text_key_skipped(self):
        chunks = [{"score": 0.9, "source": "x"}]   # no "text" key
        result = assemble(chunks, reserve_tokens=0)
        assert result == ""

    def test_empty_text_skipped(self):
        chunks = [make_chunk("")]
        result = assemble(chunks, reserve_tokens=0)
        assert result == ""


class TestAssembleSorting:
    def test_higher_score_chunk_appears_first(self):
        low  = make_chunk("low relevance chunk",  score=0.1, source="low")
        high = make_chunk("high relevance chunk", score=0.9, source="high")
        result = assemble([low, high], reserve_tokens=0)
        assert result.index("high relevance") < result.index("low relevance")

    def test_chunks_sorted_by_score_descending(self):
        chunks = [
            make_chunk("third",  score=0.3),
            make_chunk("first",  score=0.9),
            make_chunk("second", score=0.6),
        ]
        result = assemble(chunks, reserve_tokens=0)
        assert result.index("first") < result.index("second") < result.index("third")


class TestAssembleTokenBudget:
    def test_output_respects_context_limit(self):
        large_chunks = [make_chunk("word " * 300, score=float(i)) for i in range(20)]
        result = assemble(large_chunks, reserve_tokens=0)
        assert count_tokens(result) <= CONTEXT_TOKEN_LIMIT

    def test_reserve_tokens_reduces_available_budget(self):
        chunks = [make_chunk("word " * 200, score=1.0)]
        full    = assemble(chunks, reserve_tokens=0)
        reduced = assemble(chunks, reserve_tokens=500)
        assert len(reduced) <= len(full)

    def test_multiple_small_chunks_all_fit(self):
        chunks = [make_chunk(f"short chunk {i}", score=float(i)) for i in range(5)]
        result = assemble(chunks, reserve_tokens=0)
        for i in range(5):
            assert f"short chunk {i}" in result

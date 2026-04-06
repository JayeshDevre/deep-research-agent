"""
Tests for token counting and session cost tracking.
Run with: pytest tests/
"""

import pytest
from agent.budget import BudgetTracker, count_tokens, truncate_to_budget
from agent.config import (
    ANSWER_MODEL,
    DECOMPOSE_MODEL,
    SYNTH_MODEL,
    HAIKU_INPUT_COST_PER_TOKEN,
    HAIKU_OUTPUT_COST_PER_TOKEN,
    SONNET_INPUT_COST_PER_TOKEN,
    SONNET_OUTPUT_COST_PER_TOKEN,
    MAX_WEB_SEARCHES,
    SESSION_COST_LIMIT,
)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        assert count_tokens("hello") > 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hello")
        long  = count_tokens("hello world this is a longer sentence")
        assert long > short

    def test_returns_int(self):
        assert isinstance(count_tokens("test"), int)


class TestTruncateToBudget:
    def test_short_text_unchanged(self):
        text = "short"
        assert truncate_to_budget(text, max_tokens=100) == text

    def test_long_text_is_truncated(self):
        text  = "word " * 500          # ~500 tokens
        result = truncate_to_budget(text, max_tokens=10)
        assert count_tokens(result) <= 10

    def test_truncation_preserves_start(self):
        text   = "KEEP_THIS " + ("filler " * 500)
        result = truncate_to_budget(text, max_tokens=5)
        assert result.startswith("KEEP")


class TestBudgetTracker:
    def setup_method(self):
        self.budget = BudgetTracker()

    # ── Initial state ──────────────────────────────────────────────
    def test_initial_cost_is_zero(self):
        assert self.budget.session_cost == 0.0

    def test_initial_searches_is_zero(self):
        assert self.budget.web_searches_this_query == 0

    def test_can_search_initially(self):
        assert self.budget.can_search() is True

    def test_not_over_budget_initially(self):
        assert self.budget.over_session_budget() is False

    # ── Web search tracking ────────────────────────────────────────
    def test_search_count_increments(self):
        self.budget.record_web_search()
        assert self.budget.web_searches_this_query == 1

    def test_search_blocked_at_limit(self):
        for _ in range(MAX_WEB_SEARCHES):
            self.budget.record_web_search()
        assert self.budget.can_search() is False

    def test_reset_query_clears_search_count(self):
        for _ in range(MAX_WEB_SEARCHES):
            self.budget.record_web_search()
        self.budget.reset_query()
        assert self.budget.web_searches_this_query == 0
        assert self.budget.can_search() is True

    # ── Cost tracking ──────────────────────────────────────────────
    def test_haiku_answer_model_uses_haiku_pricing(self):
        self.budget.record_llm_call(1000, 100, ANSWER_MODEL)
        expected = (1000 * HAIKU_INPUT_COST_PER_TOKEN
                    + 100 * HAIKU_OUTPUT_COST_PER_TOKEN)
        assert abs(self.budget.session_cost - expected) < 1e-10

    def test_haiku_decompose_model_uses_haiku_pricing(self):
        self.budget.record_llm_call(1000, 100, DECOMPOSE_MODEL)
        expected = (1000 * HAIKU_INPUT_COST_PER_TOKEN
                    + 100 * HAIKU_OUTPUT_COST_PER_TOKEN)
        assert abs(self.budget.session_cost - expected) < 1e-10

    def test_sonnet_model_uses_sonnet_pricing(self):
        self.budget.record_llm_call(1000, 100, SYNTH_MODEL)
        expected = (1000 * SONNET_INPUT_COST_PER_TOKEN
                    + 100 * SONNET_OUTPUT_COST_PER_TOKEN)
        assert abs(self.budget.session_cost - expected) < 1e-10

    def test_sonnet_costs_more_than_haiku(self):
        haiku_budget  = BudgetTracker()
        sonnet_budget = BudgetTracker()
        haiku_budget.record_llm_call(1000, 100, ANSWER_MODEL)
        sonnet_budget.record_llm_call(1000, 100, SYNTH_MODEL)
        assert sonnet_budget.session_cost > haiku_budget.session_cost

    def test_cost_accumulates_across_calls(self):
        self.budget.record_llm_call(100, 50, ANSWER_MODEL)
        cost_after_first = self.budget.session_cost
        self.budget.record_llm_call(100, 50, ANSWER_MODEL)
        assert self.budget.session_cost > cost_after_first

    # ── Budget limits ──────────────────────────────────────────────
    def test_over_budget_when_cost_exceeds_limit(self):
        self.budget.session_cost = SESSION_COST_LIMIT + 0.001
        assert self.budget.over_session_budget() is True

    def test_budget_warning_at_80_percent(self):
        self.budget.session_cost = SESSION_COST_LIMIT * 0.81
        assert self.budget.budget_warning() is True

    def test_no_budget_warning_below_80_percent(self):
        self.budget.session_cost = SESSION_COST_LIMIT * 0.79
        assert self.budget.budget_warning() is False

    # ── Summary ────────────────────────────────────────────────────
    def test_summary_keys(self):
        summary = self.budget.summary()
        assert "session_cost_usd"     in summary
        assert "session_budget_usd"   in summary
        assert "budget_remaining_usd" in summary
        assert "total_input_tokens"   in summary
        assert "total_output_tokens"  in summary

    def test_summary_remaining_decreases_with_spend(self):
        before = self.budget.summary()["budget_remaining_usd"]
        self.budget.record_llm_call(1000, 500, ANSWER_MODEL)
        after  = self.budget.summary()["budget_remaining_usd"]
        assert after < before

"""
Tests for query decomposer — uses mocked Anthropic client to avoid API calls.
Run with: pytest tests/
"""

import pytest
from unittest.mock import MagicMock, patch
from agent.decomposer import decompose
from agent.budget import BudgetTracker


def _make_mock_client(response_text: str) -> MagicMock:
    """Build a minimal mock of anthropic.Anthropic that returns response_text."""
    content_block = MagicMock()
    content_block.text = response_text

    usage = MagicMock()
    usage.input_tokens  = 100
    usage.output_tokens = 50

    response = MagicMock()
    response.content = [content_block]
    response.usage   = usage

    client = MagicMock()
    client.messages.create.return_value = response
    return client


class TestDecomposeValidOutput:
    def setup_method(self):
        self.budget = BudgetTracker()

    def test_returns_list_of_strings(self):
        client = _make_mock_client('{"sub_questions": ["What is X?", "How does Y work?"]}')
        result = decompose("Tell me about X and Y", client, self.budget)
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    def test_correct_sub_questions_returned(self):
        client = _make_mock_client('{"sub_questions": ["What is AI?", "How is AI used?"]}')
        result = decompose("Explain AI and its uses", client, self.budget)
        assert result == ["What is AI?", "How is AI used?"]

    def test_single_sub_question_accepted(self):
        client = _make_mock_client('{"sub_questions": ["What is the capital of France?"]}')
        result = decompose("Capital of France?", client, self.budget)
        assert len(result) == 1

    def test_markdown_fenced_json_parsed(self):
        fenced = '```json\n{"sub_questions": ["Q1", "Q2"]}\n```'
        client = _make_mock_client(fenced)
        result = decompose("test query", client, self.budget)
        assert result == ["Q1", "Q2"]

    def test_budget_is_recorded(self):
        client = _make_mock_client('{"sub_questions": ["Q1"]}')
        decompose("test query", client, self.budget)
        assert self.budget.session_cost > 0
        assert self.budget.total_input_tokens > 0


class TestDecomposeInvalidOutput:
    def setup_method(self):
        self.budget = BudgetTracker()

    def test_raises_on_invalid_json(self):
        client = _make_mock_client("this is not json at all")
        with pytest.raises(ValueError, match="invalid JSON"):
            decompose("test query", client, self.budget)

    def test_raises_when_sub_questions_key_missing(self):
        client = _make_mock_client('{"questions": ["Q1"]}')
        with pytest.raises(ValueError, match="sub_questions"):
            decompose("test query", client, self.budget)

    def test_raises_when_sub_questions_empty_list(self):
        client = _make_mock_client('{"sub_questions": []}')
        with pytest.raises(ValueError, match="sub_questions"):
            decompose("test query", client, self.budget)

    def test_raises_when_sub_questions_not_list(self):
        client = _make_mock_client('{"sub_questions": "not a list"}')
        with pytest.raises(ValueError, match="sub_questions"):
            decompose("test query", client, self.budget)


class TestDecomposeUtils:
    def setup_method(self):
        self.budget = BudgetTracker()

    def test_parse_json_strips_whitespace(self):
        client = _make_mock_client('  \n  {"sub_questions": ["Q1"]}  \n  ')
        result = decompose("test", client, self.budget)
        assert result == ["Q1"]

"""
Tests for Tier 1 Working Memory (sliding window + token enforcement).
Run with: pytest tests/
"""

import pytest
from agent.memory.working import WorkingMemory
from agent.config import WORKING_MAX_TURNS, WORKING_TOKEN_LIMIT


class TestWorkingMemoryBasics:
    def setup_method(self):
        self.wm = WorkingMemory()

    def test_empty_on_init(self):
        assert self.wm.get_turns() == []

    def test_add_single_turn(self):
        self.wm.add("user", "hello")
        turns = self.wm.get_turns()
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "hello"

    def test_add_preserves_role_and_content(self):
        self.wm.add("user",      "what is AI?")
        self.wm.add("assistant", "AI is artificial intelligence.")
        turns = self.wm.get_turns()
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

    def test_clear_empties_memory(self):
        self.wm.add("user", "test")
        self.wm.clear()
        assert self.wm.get_turns() == []

    def test_as_text_formats_correctly(self):
        self.wm.add("user",      "hello")
        self.wm.add("assistant", "hi there")
        text = self.wm.as_text()
        assert "User: hello"         in text
        assert "Assistant: hi there" in text

    def test_as_text_empty_returns_empty_string(self):
        assert self.wm.as_text() == ""

    def test_token_usage_keys(self):
        usage = self.wm.token_usage()
        assert "used"  in usage
        assert "limit" in usage
        assert "turns" in usage

    def test_token_usage_turns_count(self):
        self.wm.add("user",      "question")
        self.wm.add("assistant", "answer")
        assert self.wm.token_usage()["turns"] == 2


class TestWorkingMemorySliding:
    def setup_method(self):
        self.wm = WorkingMemory()

    def test_max_turns_enforced(self):
        for i in range(WORKING_MAX_TURNS + 3):
            self.wm.add("user", f"turn {i}")
        assert len(self.wm.get_turns()) <= WORKING_MAX_TURNS

    def test_oldest_turn_evicted_first(self):
        for i in range(WORKING_MAX_TURNS + 1):
            self.wm.add("user", f"turn {i}")
        contents = [t["content"] for t in self.wm.get_turns()]
        assert "turn 0" not in contents

    def test_latest_turn_always_kept(self):
        for i in range(WORKING_MAX_TURNS + 2):
            self.wm.add("user", f"turn {i}")
        contents = [t["content"] for t in self.wm.get_turns()]
        assert f"turn {WORKING_MAX_TURNS + 1}" in contents


class TestWorkingMemoryTokenLimit:
    def setup_method(self):
        self.wm = WorkingMemory()

    def test_token_usage_does_not_exceed_limit(self):
        long_text = "word " * 300   # ~300 tokens per turn
        for _ in range(5):
            self.wm.add("user", long_text)
        assert self.wm.token_usage()["used"] <= WORKING_TOKEN_LIMIT

    def test_always_keeps_at_least_one_turn(self):
        # Even a single oversized turn must not cause infinite eviction loop
        huge_text = "token " * 1000
        self.wm.add("user", huge_text)
        assert len(self.wm.get_turns()) == 1

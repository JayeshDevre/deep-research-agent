"""
Token counter and session cost tracker.
Enforces hard budget limits defined in config.py.
"""

import threading
from typing import Any

import tiktoken

from agent.config import (
    CONTEXT_TOKEN_LIMIT,
    SESSION_COST_LIMIT,
    MAX_WEB_SEARCHES,
    HAIKU_INPUT_COST_PER_TOKEN,
    HAIKU_OUTPUT_COST_PER_TOKEN,
    SONNET_INPUT_COST_PER_TOKEN,
    SONNET_OUTPUT_COST_PER_TOKEN,
    ANSWER_MODEL,
    DECOMPOSE_MODEL,
)
from agent.utils import get_logger

logger = get_logger(__name__)

try:
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception as exc:
    raise RuntimeError(
        "tiktoken encoding 'cl100k_base' could not be loaded. "
        "Run: pip install tiktoken"
    ) from exc

# Models that use Haiku pricing — extend if new Haiku variants are added.
_HAIKU_MODEL_IDS = {ANSWER_MODEL, DECOMPOSE_MODEL}


def _cost_for_model(model: str, input_tokens: int, output_tokens: int) -> float:
    if model in _HAIKU_MODEL_IDS:
        return (
            input_tokens  * HAIKU_INPUT_COST_PER_TOKEN
            + output_tokens * HAIKU_OUTPUT_COST_PER_TOKEN
        )
    return (
        input_tokens  * SONNET_INPUT_COST_PER_TOKEN
        + output_tokens * SONNET_OUTPUT_COST_PER_TOKEN
    )


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def truncate_to_budget(text: str, max_tokens: int) -> str:
    tokens = _encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoder.decode(tokens[:max_tokens])


class BudgetTracker:
    """Thread-safe token counter and cost tracker.

    The agent answers sub-questions in parallel threads, so all mutable
    state is guarded by a single lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.session_cost: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.web_searches_this_query: int = 0

    def reset_query(self) -> None:
        with self._lock:
            self.web_searches_this_query = 0

    def record_llm_call(self, input_tokens: int, output_tokens: int, model: str) -> None:
        cost = _cost_for_model(model, input_tokens, output_tokens)
        with self._lock:
            self.session_cost += cost
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
        logger.debug(
            "LLM call | model=%s in=%d out=%d cost=$%.5f total=$%.5f",
            model, input_tokens, output_tokens, cost, self.session_cost,
        )

    def claim_search(self) -> bool:
        """Atomically check and reserve a web search slot.

        Returns True if a slot was available and has been reserved,
        False if the limit was already reached. Combines the check and
        increment in one lock acquisition to eliminate the TOCTOU race
        when sub-questions run in parallel threads.
        """
        with self._lock:
            if self.web_searches_this_query < MAX_WEB_SEARCHES:
                self.web_searches_this_query += 1
                logger.debug("Web search #%d this query", self.web_searches_this_query)
                return True
            return False

    # Keep for backward-compat (n8n health endpoint reads this via /health)
    def can_search(self) -> bool:
        with self._lock:
            return self.web_searches_this_query < MAX_WEB_SEARCHES

    def over_session_budget(self) -> bool:
        with self._lock:
            return self.session_cost >= SESSION_COST_LIMIT

    def budget_warning(self) -> bool:
        """True when 80% of session budget is consumed."""
        with self._lock:
            return self.session_cost >= SESSION_COST_LIMIT * 0.8

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_cost_usd":      round(self.session_cost, 5),
                "session_budget_usd":    SESSION_COST_LIMIT,
                "budget_remaining_usd":  round(SESSION_COST_LIMIT - self.session_cost, 5),
                "total_input_tokens":    self.total_input_tokens,
                "total_output_tokens":   self.total_output_tokens,
                "context_token_limit":   CONTEXT_TOKEN_LIMIT,
            }

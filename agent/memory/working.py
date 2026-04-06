"""
Tier 1 — Working Memory (in-process sliding window).

Keeps the last N conversation turns within a token budget.
Oldest turns are evicted when the token limit is exceeded.
"""

from collections import deque
from typing import Literal

from agent.budget import count_tokens
from agent.config import WORKING_MAX_TURNS, WORKING_TOKEN_LIMIT
from agent.utils import get_logger

logger = get_logger(__name__)

Role = Literal["user", "assistant"]
_ROLE_LABEL: dict[Role, str] = {"user": "User", "assistant": "Assistant"}


class WorkingMemory:
    def __init__(self) -> None:
        self._turns: deque[dict[str, str]] = deque(maxlen=WORKING_MAX_TURNS)

    def add(self, role: Role, content: str) -> None:
        """Add a turn and enforce the token limit via FIFO eviction."""
        self._turns.append({"role": role, "content": content})
        self._enforce_token_limit()

    def _enforce_token_limit(self) -> None:
        evicted = 0
        while self._token_count() > WORKING_TOKEN_LIMIT and len(self._turns) > 1:
            self._turns.popleft()
            evicted += 1
        if evicted:
            logger.debug("Evicted %d turn(s) from working memory", evicted)

    def _token_count(self) -> int:
        return sum(count_tokens(t["content"]) for t in self._turns)

    def get_turns(self) -> list[dict[str, str]]:
        return list(self._turns)

    def as_text(self) -> str:
        lines = [
            f"{_ROLE_LABEL.get(t['role'], t['role'])}: {t['content']}"
            for t in self._turns
        ]
        return "\n".join(lines)

    def token_usage(self) -> dict[str, int]:
        used = self._token_count()
        return {"used": used, "limit": WORKING_TOKEN_LIMIT, "turns": len(self._turns)}

    def clear(self) -> None:
        self._turns.clear()
        logger.debug("Working memory cleared")

"""
Context assembler: packs retrieved chunks into the token budget.
Sorts by relevance score (descending) and hard-stops at CONTEXT_TOKEN_LIMIT.
"""

from typing import Any

from agent.budget import count_tokens
from agent.config import CONTEXT_TOKEN_LIMIT, CHARS_PER_TOKEN_ESTIMATE
from agent.utils import get_logger

logger = get_logger(__name__)


def assemble(chunks: list[dict[str, Any]], reserve_tokens: int = 200) -> str:
    """
    Pack chunks into a single context string within the token budget.

    Args:
        chunks:         List of {"text": str, "source": str, "score": float}.
                        Missing keys are handled gracefully.
        reserve_tokens: Tokens reserved for system prompt + user query overhead.
                        Set to 0 when the assembled string is a tool result
                        (not the final LLM context).

    Returns:
        A single string ready to insert into an LLM prompt.
    """
    if not chunks:
        return ""

    budget = CONTEXT_TOKEN_LIMIT - reserve_tokens
    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)

    assembled: list[str] = []
    used = 0

    for chunk in sorted_chunks:
        source = chunk.get("source", "unknown")
        body   = str(chunk.get("text", "")).strip()
        if not body:
            continue

        header = f"[Source: {source}]\n"
        block  = header + body + "\n\n"
        tokens = count_tokens(block)

        if used + tokens > budget:
            remaining = budget - used
            if remaining > 50:
                max_chars = remaining * CHARS_PER_TOKEN_ESTIMATE
                assembled.append(header + body[:max_chars] + "…")
                logger.debug("Truncated chunk from %s (%d tokens over budget)", source, tokens - remaining)
            break

        assembled.append(block)
        used += tokens

    logger.debug("Assembled %d/%d chunks | %d tokens used", len(assembled), len(chunks), used)
    return "".join(assembled).strip()

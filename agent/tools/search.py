"""
Web search tool — Tavily wrapper.
Returns clean, token-bounded chunks ready for the context assembler.
"""

from typing import Any

from tavily import TavilyClient

from agent.config import SEARCH_MAX_RESULTS, SEARCH_MAX_CHARS, SEARCH_DEFAULT_SCORE
from agent.utils import get_logger

logger = get_logger(__name__)


def search(query: str, client: TavilyClient) -> list[dict[str, Any]]:
    """
    Search the web via Tavily and return result chunks.

    Args:
        query:  The search query string.
        client: An initialised TavilyClient.

    Returns:
        List of {"text": str, "source": str, "score": float}.
        Returns empty list on API failure (non-fatal).
    """
    logger.info("Web search: %s", query[:80])

    try:
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=SEARCH_MAX_RESULTS,
            include_answer=False,
        )
    except Exception:
        logger.exception("Tavily search failed for query='%s'", query[:80])
        return []

    chunks: list[dict[str, Any]] = []
    for result in response.get("results", []):
        body = (result.get("content") or result.get("snippet") or "").strip()
        if not body:
            logger.debug("Skipping result with no content: %s", result.get("url"))
            continue
        chunks.append({
            "text":   body[:SEARCH_MAX_CHARS],
            "source": result.get("url", "web"),
            "score":  result.get("score", SEARCH_DEFAULT_SCORE),
        })

    logger.info("Web search returned %d chunks", len(chunks))
    return chunks

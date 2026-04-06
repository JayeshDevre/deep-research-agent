"""
Query decomposer: breaks a complex user question into focused sub-questions.
Uses a cheap/fast model since this is a structured extraction task.
"""

import anthropic

from agent.budget import BudgetTracker
from agent.config import DECOMPOSE_MODEL
from agent.utils import get_logger, parse_json_response

logger = get_logger(__name__)

_PROMPT = """You are a research planner. Break the user's question into 2-4 focused sub-questions that together fully answer the original query.

Rules:
- Each sub-question must be self-contained and directly searchable
- Prefer specific sub-questions over vague ones
- If the question is already simple, return exactly 1 sub-question
- Return ONLY valid JSON — no explanation, no markdown prose

Output format:
{{"sub_questions": ["question 1", "question 2", ...]}}

User question: {query}"""


def decompose(
    query: str,
    client: anthropic.Anthropic,
    budget: BudgetTracker,
) -> list[str]:
    """
    Decompose a complex query into 1-4 focused sub-questions.

    Returns:
        List of sub-question strings.

    Raises:
        ValueError: if the LLM response cannot be parsed as expected JSON.
        anthropic.APIError: on API-level failures (propagated to caller).
    """
    logger.info("Decomposing query: %s", query[:80])

    response = client.messages.create(
        model=DECOMPOSE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
    )
    budget.record_llm_call(
        response.usage.input_tokens,
        response.usage.output_tokens,
        DECOMPOSE_MODEL,
    )

    raw = response.content[0].text.strip()

    try:
        data = parse_json_response(raw)
    except Exception as exc:
        raise ValueError(
            f"Decomposer returned invalid JSON: {raw[:200]}"
        ) from exc

    sub_questions = data.get("sub_questions")
    if not isinstance(sub_questions, list) or not sub_questions:
        raise ValueError(
            f"Decomposer JSON missing 'sub_questions' list: {data}"
        )

    logger.info("Decomposed into %d sub-questions", len(sub_questions))
    return sub_questions

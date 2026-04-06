"""
Main Research Agent — tool-use architecture.

Claude decides when to call web_search vs query_memory.
Python enforces budget constraints and executes the tools.

Flow per query:
  1. Decompose user question → sub-questions (claude-haiku)
  2. For each sub-question, run tool-use loop:
       Claude calls query_memory and/or web_search as needed
       Python enforces budget before executing each tool
       Loop ends when Claude emits end_turn
  3. Synthesise partial answers → final answer (claude-sonnet)
  4. Extract facts from final answer → store in episodic memory
  5. Add Q+A to working memory

Flow per session end:
  Summarise full session → persist to episodic memory
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import anthropic
from dotenv import load_dotenv
from tavily import TavilyClient

from agent.budget import BudgetTracker
from agent.config import (
    ANSWER_MODEL,
    SYNTH_MODEL,
    CONTEXT_TOKEN_LIMIT,
    SESSION_COST_LIMIT,
    MAX_WEB_SEARCHES,
    MAX_TOOL_ITERATIONS,
)
from agent.context_assembler import assemble
from agent.decomposer import decompose
from agent.memory.episodic import EpisodicMemory
from agent.memory.working import WorkingMemory
from agent.tools.search import search as tavily_search
from agent.utils import get_logger, parse_json_response

load_dotenv()
logger = get_logger(__name__)

# ------------------------------------------------------------------
# Tool definitions — passed to Claude on every sub-question call
# ------------------------------------------------------------------
_TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information on a topic. "
            "Use this when memory does not contain relevant information "
            "or when the question requires recent data. "
            f"Limited to {MAX_WEB_SEARCHES} calls per query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_memory",
        "description": (
            "Search past research sessions and extracted facts stored in memory. "
            "Always try this before web_search — it is free and instant. "
            "Returns relevant chunks from previous sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search memory for.",
                }
            },
            "required": ["query"],
        },
    },
]

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a precise research assistant operating under strict constraints.

CONSTRAINTS:
- Max context tokens per call: {context_limit}
- Max session cost: ${session_cost_limit}
- Web searches remaining this query: {searches_remaining}

TOOL STRATEGY:
1. Always call query_memory first (free, instant)
2. Only call web_search if memory is insufficient
3. Stop calling tools once you have enough information

RECENT CONVERSATION:
{working_memory}

Answer concisely in 3-5 sentences. If information is insufficient after using tools, say so explicitly."""

_SYNTH_PROMPT = """Combine these partial answers into one coherent, well-structured response.

Original question: {query}

Partial answers:
{partials}

Write a clear, comprehensive answer. Use bullet points for lists. Be concise."""

_FACT_EXTRACT_PROMPT = """Extract 3-5 key facts from the text as short standalone sentences.
Return ONLY a JSON array of strings. No explanation, no markdown.

Text: {text}"""

_SESSION_SUMMARY_PROMPT = """Summarise this research session in 3-5 sentences, capturing the main topics and key findings.

Session:
{session_text}"""


class ResearchAgent:
    def __init__(self) -> None:
        api_key    = os.getenv("ANTHROPIC_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        if not tavily_key:
            raise EnvironmentError("TAVILY_API_KEY is not set.")

        self.anthropic = anthropic.Anthropic(api_key=api_key)
        self.tavily    = TavilyClient(api_key=tavily_key)
        self.budget    = BudgetTracker()
        self.working   = WorkingMemory()
        self.episodic  = EpisodicMemory()

        self._session_log:  list[str] = []
        self._sources_used: list[str] = []

        logger.info("ResearchAgent initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, user_query: str) -> dict[str, Any]:
        """
        Research a complex question and return a structured result.

        Returns dict with keys:
            answer, sub_questions, sources_used, memory_hits,
            web_searches, tool_calls, trace, budget
        """
        if self.budget.over_session_budget():
            logger.warning("Session budget exhausted")
            return {
                "answer": "Session budget exhausted ($0.05 limit reached). Start a new session.",
                "sub_questions": [], "sources_used": [], "memory_hits": 0,
                "web_searches": 0, "tool_calls": 0, "trace": [], "budget": self.budget.summary(),
            }

        self.budget.reset_query()
        self._sources_used = []
        total_tool_calls = 0
        memory_hits = 0

        logger.info("Query: %s", user_query[:100])

        # Step 1: Decompose
        try:
            sub_questions = decompose(user_query, self.anthropic, self.budget)
        except Exception:
            logger.exception("Decomposition failed")
            sub_questions = [user_query]   # fall back to treating query as single sub-question

        self._log(f"Q: {user_query}")

        # Step 2: Answer sub-questions in parallel (one thread per sub-question)
        partial_answers: list[str]           = [""] * len(sub_questions)
        full_trace:      list[dict[str, Any]] = [{}] * len(sub_questions)

        with ThreadPoolExecutor(max_workers=len(sub_questions)) as pool:
            futures = {
                pool.submit(self._answer_with_tools, sq): i
                for i, sq in enumerate(sub_questions)
            }
            for future in as_completed(futures):
                i = futures[future]
                sq = sub_questions[i]
                answer, tool_calls, mem_hits, trace = future.result()
                partial_answers[i] = f"Sub-question: {sq}\nAnswer: {answer}"
                total_tool_calls  += tool_calls
                memory_hits       += mem_hits
                full_trace[i]      = {"sub_question": sq, "answer": answer, "trace": trace}

        # Step 3: Synthesise
        final_answer = self._synthesise(user_query, partial_answers)

        # Step 4: Extract + store facts (best-effort)
        self._extract_and_store_facts(final_answer)

        # Step 5: Update working memory
        self.working.add("user",      user_query)
        self.working.add("assistant", final_answer)
        self._log(f"A: {final_answer}")

        return {
            "answer":        final_answer,
            "sub_questions": sub_questions,
            "sources_used":  list(set(self._sources_used)),
            "memory_hits":   memory_hits,
            "web_searches":  self.budget.web_searches_this_query,
            "tool_calls":    total_tool_calls,
            "trace":         full_trace,
            "budget":        self.budget.summary(),
        }

    def end_session(self) -> str:
        """Summarise and persist the current session to episodic memory."""
        if not self._session_log:
            return "No session content to save."

        session_text = "\n".join(self._session_log)
        prompt = _SESSION_SUMMARY_PROMPT.format(session_text=session_text)

        try:
            response = self.anthropic.messages.create(
                model=ANSWER_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.content[0].text.strip()
            self.budget.record_llm_call(
                response.usage.input_tokens,
                response.usage.output_tokens,
                ANSWER_MODEL,
            )
        except Exception:
            logger.exception("Session summarisation failed")
            summary = "Session saved (summary unavailable)."

        self.episodic.store_session_summary(summary)
        self.working.clear()
        self._session_log = []
        logger.info("Session ended and saved to episodic memory")
        return summary

    # ------------------------------------------------------------------
    # Agentic tool-use loop
    # ------------------------------------------------------------------

    def _answer_with_tools(
        self, sub_question: str
    ) -> tuple[str, int, int, list[dict[str, Any]]]:
        """
        Run the tool-use loop for one sub-question.

        Returns:
            (answer_text, tool_call_count, memory_hit_count, tool_trace)
        """
        system = _SYSTEM_PROMPT.format(
            context_limit=CONTEXT_TOKEN_LIMIT,
            session_cost_limit=SESSION_COST_LIMIT,
            searches_remaining=MAX_WEB_SEARCHES - self.budget.web_searches_this_query,
            working_memory=self.working.as_text() or "None yet.",
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": sub_question}]
        tool_call_count  = 0
        memory_hit_count = 0
        tool_trace:    list[dict[str, Any]] = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = self.anthropic.messages.create(
                    model=ANSWER_MODEL,
                    max_tokens=500,
                    system=system,
                    tools=_TOOLS,
                    messages=messages,
                )
            except anthropic.APIError:
                logger.exception("Anthropic API error on iteration %d", iteration)
                break

            self.budget.record_llm_call(
                response.usage.input_tokens,
                response.usage.output_tokens,
                ANSWER_MODEL,
            )

            if response.stop_reason == "end_turn":
                answer = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                ).strip()
                return answer, tool_call_count, memory_hit_count, tool_trace

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results: list[dict[str, Any]] = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_call_count += 1
                    result_text = self._execute_tool(block.name, block.input)
                    hit = block.name == "query_memory" and "No results" not in result_text

                    if hit:
                        memory_hit_count += 1

                    tool_trace.append({
                        "tool":  block.name,
                        "query": block.input.get("query", ""),
                        "hit":   hit,
                    })
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_text,
                    })

                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason
            logger.warning("Unexpected stop_reason='%s'", response.stop_reason)
            break

        # Loop ended without end_turn — extract last assistant text if any
        return (
            self._extract_last_text(messages),
            tool_call_count,
            memory_hit_count,
            tool_trace,
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> str:
        query = inputs.get("query", "")
        if name == "web_search":
            return self._run_web_search(query)
        if name == "query_memory":
            return self._run_query_memory(query)
        logger.warning("Unknown tool requested: %s", name)
        return f"Unknown tool: {name}"

    def _run_web_search(self, query: str) -> str:
        if not self.budget.can_search():
            logger.info("Web search blocked — limit reached")
            return f"CONSTRAINT: Web search limit reached (max {MAX_WEB_SEARCHES} per query). Use available context."

        if self.budget.over_session_budget():
            return "CONSTRAINT: Session cost budget exhausted."

        chunks = tavily_search(query, self.tavily)
        self.budget.record_web_search()

        if not chunks:
            return "No results found."

        self._sources_used.extend(c["source"] for c in chunks)
        return assemble(chunks, reserve_tokens=0) or "No usable content returned."

    def _run_query_memory(self, query: str) -> str:
        chunks = self.episodic.query(query)
        if not chunks:
            return "No results found in memory."
        return assemble(chunks, reserve_tokens=0) or "No usable content in memory."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _synthesise(self, query: str, partial_answers: list[str]) -> str:
        prompt = _SYNTH_PROMPT.format(
            query=query,
            partials="\n\n".join(partial_answers),
        )
        try:
            response = self.anthropic.messages.create(
                model=SYNTH_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            self.budget.record_llm_call(
                response.usage.input_tokens,
                response.usage.output_tokens,
                SYNTH_MODEL,
            )
            return response.content[0].text.strip()
        except anthropic.APIError:
            logger.exception("Synthesis API call failed")
            # Fall back to joining partial answers directly
            return "\n\n".join(partial_answers)

    def _extract_and_store_facts(self, text: str) -> None:
        """Extract key facts from an answer and store them in episodic memory."""
        prompt = _FACT_EXTRACT_PROMPT.format(text=text[:1_500])
        try:
            response = self.anthropic.messages.create(
                model=ANSWER_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            self.budget.record_llm_call(
                response.usage.input_tokens,
                response.usage.output_tokens,
                ANSWER_MODEL,
            )
            facts = parse_json_response(response.content[0].text)
            if isinstance(facts, list):
                self.episodic.store_facts_bulk(facts)
        except Exception:
            # Fact extraction is best-effort — never crash the main flow
            logger.debug("Fact extraction skipped (non-fatal)", exc_info=True)

    def _extract_last_text(self, messages: list[dict[str, Any]]) -> str:
        """Pull the last assistant text from a messages list after loop exhaustion."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg["content"]
                if isinstance(content, list):
                    text = next((b.text for b in content if hasattr(b, "text")), None)
                    if text:
                        return text.strip()
                elif isinstance(content, str):
                    return content.strip()
        return "No answer could be generated within the tool call limit."

    def _log(self, line: str) -> None:
        self._session_log.append(line)

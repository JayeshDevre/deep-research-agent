"""
FastAPI wrapper around ResearchAgent.
Exposes the agent over HTTP so n8n (or any external orchestrator) can call it.

Endpoints:
    GET  /health          — agent status (budget, memory count)
    POST /query           — full research pipeline (decompose → tools → synthesise)
    POST /memory/search   — search memory only (no web, no LLM cost) — used by n8n
    GET  /memory/stats    — detailed memory tier breakdown — used by n8n routing
    POST /session/end     — summarise + persist session, reset agent

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.agent import ResearchAgent
from agent.context_assembler import assemble
from agent.utils import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Singleton agent (persists memory across requests in the same process) ──
_agent: ResearchAgent | None = None


def get_agent() -> ResearchAgent:
    global _agent
    if _agent is None:
        _agent = ResearchAgent()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising ResearchAgent")
    get_agent()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Deep Research Agent API",
    description="Memory-constrained research agent with 3-tier memory and Claude native tool use.",
    version="1.0.0",
    lifespan=lifespan,
)

_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5678").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Request / Response models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The research question to answer.")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The query to search memory for.")
    n_results: int = Field(default=4, ge=1, le=10, description="Max chunks to return.")


class QueryResponse(BaseModel):
    answer:        str
    sub_questions: list[str]
    sources_used:  list[str]
    memory_hits:   int
    web_searches:  int
    tool_calls:    int
    budget:        dict[str, Any]


class MemorySearchResponse(BaseModel):
    has_results:    bool
    num_chunks:     int
    best_score:     float
    assembled_text: str
    chunks:         list[dict[str, Any]]


class MemoryStatsResponse(BaseModel):
    total_chunks:       int
    total_facts:        int
    working_memory:     dict[str, Any]
    budget:             dict[str, Any]
    memory_warm:        bool


class SessionEndResponse(BaseModel):
    summary: str


class HealthResponse(BaseModel):
    status:       str
    memory_chunks: int
    session_cost:  float
    searches_left: int


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Returns current agent status — useful for n8n IF node checks."""
    agent  = get_agent()
    budget = agent.budget.summary()
    from agent.config import MAX_WEB_SEARCHES
    return {
        "status":        "ok",
        "memory_chunks": agent.episodic.count(),
        "session_cost":  budget["session_cost_usd"],
        "searches_left": MAX_WEB_SEARCHES - agent.budget.web_searches_this_query,
    }


@app.post("/memory/search", response_model=MemorySearchResponse)
def memory_search(request: MemorySearchRequest):
    """
    Search episodic + fact memory without triggering web search or LLM calls.
    n8n uses this to decide whether the query can be answered from memory alone.
    Zero cost — only a ChromaDB vector lookup.
    """
    agent  = get_agent()
    chunks = agent.episodic.query(request.query, n=request.n_results)

    best_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
    assembled  = assemble(chunks, reserve_tokens=0) if chunks else ""

    return {
        "has_results":    len(chunks) > 0,
        "num_chunks":     len(chunks),
        "best_score":     round(best_score, 4),
        "assembled_text": assembled,
        "chunks":         chunks,
    }


@app.get("/memory/stats", response_model=MemoryStatsResponse)
def memory_stats():
    """
    Detailed memory tier breakdown for n8n routing decisions.
    memory_warm=true when episodic memory has >=5 chunks (past the cold-start phase).
    """
    agent       = get_agent()
    total       = agent.episodic.count()
    total_facts = agent.episodic.count_facts()

    return {
        "total_chunks":   total,
        "total_facts":    total_facts,
        "working_memory": agent.working.token_usage(),
        "budget":         agent.budget.summary(),
        "memory_warm":    total >= 5,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Run a research query through the full agent pipeline:
      decompose → tool-use loop → synthesise → store facts
    """
    agent = get_agent()

    if agent.budget.over_session_budget():
        raise HTTPException(
            status_code=402,
            detail="Session budget exhausted. Call POST /session/end to reset.",
        )

    result = agent.query(request.query)
    return {
        "answer":        result["answer"],
        "sub_questions": result["sub_questions"],
        "sources_used":  result["sources_used"],
        "memory_hits":   result["memory_hits"],
        "web_searches":  result["web_searches"],
        "tool_calls":    result["tool_calls"],
        "budget":        result["budget"],
    }


@app.post("/session/end", response_model=SessionEndResponse)
def end_session():
    """
    Summarise and persist the current session to episodic memory.
    Call this when a user finishes a research session.
    """
    agent   = get_agent()
    summary = agent.end_session()
    global _agent
    _agent  = None
    return {"summary": summary}

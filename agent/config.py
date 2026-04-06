"""
Central configuration for the Deep Research Agent.
All constants and tuneable parameters live here.
Change values here to affect the whole system — no hunting through files.
"""

# ------------------------------------------------------------------
# Models
# Haiku for cheap/fast tasks; Sonnet only for final synthesis.
# ------------------------------------------------------------------
DECOMPOSE_MODEL = "claude-haiku-4-5-20251001"
ANSWER_MODEL    = "claude-haiku-4-5-20251001"
SYNTH_MODEL     = "claude-sonnet-4-6"

# ------------------------------------------------------------------
# Budget constraints (self-imposed, documented in evaluation.md)
# ------------------------------------------------------------------
CONTEXT_TOKEN_LIMIT = 2_000   # max tokens packed into a single LLM call
SESSION_COST_LIMIT  = 0.05    # max USD spend per session
MAX_WEB_SEARCHES    = 3       # max Tavily calls per query

# Haiku pricing per token (Anthropic, April 2026)
HAIKU_INPUT_COST_PER_TOKEN  = 0.80  / 1_000_000
HAIKU_OUTPUT_COST_PER_TOKEN = 4.00  / 1_000_000

# Sonnet pricing per token (Anthropic, April 2026)
SONNET_INPUT_COST_PER_TOKEN  = 3.00  / 1_000_000
SONNET_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000

# ------------------------------------------------------------------
# Working memory (Tier 1)
# ------------------------------------------------------------------
WORKING_MAX_TURNS   = 5     # sliding window size
WORKING_TOKEN_LIMIT = 800   # token slice reserved for recent turns

# ------------------------------------------------------------------
# Episodic memory (Tier 2 + 3)
# ------------------------------------------------------------------
CHROMA_COLLECTION = "research_memory"
CHROMA_N_RESULTS  = 4     # chunks retrieved per semantic query

# ------------------------------------------------------------------
# Web search tool
# ------------------------------------------------------------------
SEARCH_MAX_RESULTS = 3    # Tavily results per call
SEARCH_MAX_CHARS   = 800  # truncate each result body to limit tokens
SEARCH_DEFAULT_SCORE = 0.5  # fallback relevance score when not provided

# ------------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------------
MAX_TOOL_ITERATIONS = 4   # max tool calls per sub-question (loop guard)

# ------------------------------------------------------------------
# Context assembler
# ------------------------------------------------------------------
# Rough chars-per-token estimate used when truncating partial chunks.
# Tokens average ~4 chars in English; conservative to avoid overflow.
CHARS_PER_TOKEN_ESTIMATE = 4

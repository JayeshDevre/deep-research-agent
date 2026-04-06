# Deep Research Agent

A memory-constrained AI research agent that answers complex multi-part questions intelligently — checking its own memory before searching the web, and operating within strict token and cost budgets.

## Architecture

```
User Query
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    n8n Workflow (Orchestration)              │
│                                                             │
│  Webhook → Validate → Health Check → Budget Gate            │
│                                         ↓                   │
│                                  Search Memory (POST /memory/search)
│                                         ↓                   │
│                              ┌── Memory Sufficient? ──┐     │
│                              │  (score ≥ 0.40,        │     │
│                              │   ≥ 2 chunks)          │     │
│                          YES ↓                    NO  ↓     │
│                   Return Memory Answer    Run Full Research  │
│                                                 ↓           │
│                                         Budget > 80%?       │
│                                        YES ↓      NO ↓     │
│                                  Auto-Save Session  Return  │
└─────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │       FastAPI (port 8000)    │
                    └────────────┬────────────────┘
                                 │
Full Research Pipeline           ▼
    ↓
Query Decomposer          (claude-haiku → 2-4 sub-questions)
    ↓
Tool-Use Loop (per sub-question)
    Claude decides which tool to call:
    ├── query_memory → Tier 1: Working Memory  (last 5 turns, 800 token cap)
    │                  Tier 2: Episodic Memory  (ChromaDB — session summaries)
    │                  Tier 3: Semantic Memory  (ChromaDB — extracted facts)
    └── web_search   → Tavily (max 3 calls/query, budget-gated)
    ↓
Context Assembler         (packs chunks within 2,000 token hard limit)
    ↓
Partial Answer            (claude-haiku per sub-question)
    ↓
Synthesis                 (claude-sonnet — final coherent answer)
    ↓
Fact Extraction + Memory Write
```

## Self-Imposed Constraints

| Constraint | Value | Reason |
|---|---|---|
| Max context tokens per LLM call | 2,000 | Forces selective retrieval |
| Max session cost | $0.05 | Simulates real client budget limits |
| Max web searches per query | 3 | Forces memory reuse over always searching |
| Working memory turns | 5 | Prevents unbounded context growth |

## Setup

### 1. Clone & install

```bash
git clone <repo-url>
cd deep-research-agent
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add API keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# ANTHROPIC_API_KEY=...
# TAVILY_API_KEY=...
```

Get keys:
- Anthropic: https://console.anthropic.com
- Tavily: https://tavily.com (free tier available)

### 3. Run the Streamlit UI

```bash
streamlit run ui/app.py
```

Open http://localhost:8501

### 4. Run the FastAPI server (for n8n)

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at http://localhost:8000/docs

### 5. Import the n8n workflow

1. Start n8n via Docker:
   ```bash
   docker run -it --rm --name n8n -p 5678:5678 -v ~/.n8n:/home/node/.n8n n8nio/n8n
   ```
2. Open `http://localhost:5678`
3. Go to **Workflows → Import from file**
4. Select `n8n_workflow.json`
5. Activate the workflow
6. Send a test request:

```bash
curl -X POST http://localhost:5678/webhook/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What caused the 2008 financial crisis?"}'
```

**n8n workflow handles both query routing and memory management:**

| Step | Node | What it does |
|------|------|-------------|
| 1 | Receive Query | Webhook receives POST with `{ "query": "..." }` |
| 2 | Query Provided? | Validates query is not empty (400 if missing) |
| 3 | Check Agent Health | `GET /health` — reads session cost, memory count |
| 4 | Budget Available? | Gates on budget < $0.05 (402 if exhausted) |
| 5 | **Search Memory** | `POST /memory/search` — queries ChromaDB for existing knowledge (zero cost) |
| 6 | **Memory Sufficient?** | Routes based on memory quality: cosine score ≥ 0.40 AND ≥ 2 chunks |
| 7a | Return Memory Answer | If memory is sufficient → respond directly (no LLM/web cost) |
| 7b | Run Full Research | If memory insufficient → `POST /query` (full pipeline) |
| 8 | **Budget > 80%?** | Post-query: checks if session cost ≥ $0.04 |
| 9 | **Auto-Save Session** | If budget high → `POST /session/end` (persists to episodic memory) |
| 10 | Return Result | Returns the research answer with budget info |

**Key routing decisions made by n8n:**
- **Memory-first routing**: queries memory before invoking the expensive research pipeline. Over time, as episodic memory grows, more queries are answered from memory alone (zero web search cost).
- **Automatic session management**: when budget exceeds 80%, n8n auto-saves the session to episodic memory, ensuring knowledge is persisted before budget exhaustion.

**API endpoints used by n8n:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Budget and status check |
| `/memory/search` | POST | Pre-query memory lookup (routing decision) |
| `/memory/stats` | GET | Detailed memory tier info (optional monitoring) |
| `/query` | POST | Full research pipeline |
| `/session/end` | POST | Session save + reset (memory management) |

> **Note:** The workflow uses `host.docker.internal:8000` (Docker's alias for the host machine). If running n8n natively (not Docker), replace with `127.0.0.1:8000` in each HTTP Request node.

### 6. Run tests

```bash
pytest tests/ -v
```

57 tests covering budget tracking, working memory, context assembler, and query decomposer.

## Project Structure

```
deep-research-agent/
├── agent/
│   ├── agent.py               # main orchestration loop
│   ├── decomposer.py          # query breakdown (haiku)
│   ├── context_assembler.py   # token-budget packing
│   ├── budget.py              # token counter + cost tracker
│   ├── config.py              # all constants and tuneable parameters
│   ├── utils.py               # shared logging + JSON parsing
│   ├── memory/
│   │   ├── working.py         # Tier 1: sliding window buffer
│   │   └── episodic.py        # Tier 2+3: ChromaDB vector store
│   └── tools/
│       └── search.py          # Tavily web search wrapper
├── api/
│   └── main.py                # FastAPI server (/query, /memory/search, /memory/stats, /session/end)
├── ui/
│   └── app.py                 # Streamlit demo
├── tests/
│   ├── test_budget.py
│   ├── test_working_memory.py
│   ├── test_context_assembler.py
│   └── test_decomposer.py
├── n8n_workflow.json          # importable n8n workflow
├── evaluation.md              # architecture trade-off analysis
├── requirements.txt
└── README.md
```

## Example Queries

- "What are the economic impacts of AI on developing countries and what policies are being proposed?"
- "How does RAG compare to fine-tuning for enterprise LLM applications, and when should each be used?"
- "What caused the 2023 banking crisis and how does it compare to 2008?"

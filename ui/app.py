"""
Streamlit UI — Deep Research Agent Dashboard
Run with: streamlit run ui/app.py
"""

import sys
import os
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from agent.agent import ResearchAgent
from agent.config import MAX_WEB_SEARCHES, SESSION_COST_LIMIT, CONTEXT_TOKEN_LIMIT

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Base ── */
.stApp { background: #0f172a; }
section[data-testid="stSidebar"] { background: #111827 !important; border-right: 1px solid #1e293b; }

/* ── Page title ── */
.dash-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.dash-sub {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 24px;
}
.dash-sub b { color: #94a3b8; font-weight: 500; }

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    vertical-align: middle;
}
.status-ready    { background: #052e16; border: 1px solid #166534; color: #4ade80; }
.status-warning  { background: #1c1500; border: 1px solid #713f12; color: #fbbf24; }
.status-exceeded { background: #1c0505; border: 1px solid #7f1d1d; color: #f87171; }

/* ── Warning banner ── */
.warn-banner {
    background: #1c1500;
    border: 1px solid #713f12;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.95rem;
    color: #fbbf24;
    font-weight: 500;
}
.exceeded-banner {
    background: #1c0505;
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 20px;
    font-size: 0.95rem;
    color: #f87171;
    font-weight: 500;
}

/* ── Query box ── */
.query-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 8px;
}
textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-size: 0.97rem !important;
    line-height: 1.6 !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea::placeholder { color: #475569 !important; font-size: 1rem !important; }
textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important; }

/* ── Submit button ── */
[data-testid="stFormSubmitButton"] button {
    background: #3b82f6 !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    padding: 12px 20px !important;
    width: 100%;
    transition: background 0.15s !important;
}
[data-testid="stFormSubmitButton"] button:hover { background: #2563eb !important; }

/* ── Example query cards ── */
.example-section-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
    margin-top: 32px;
}
.stButton > button.example-btn {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    font-size: 0.92rem !important;
    font-weight: 400 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    text-align: left !important;
    line-height: 1.5 !important;
    transition: border-color 0.15s, color 0.15s !important;
    width: 100%;
}
.stButton > button.example-btn:hover {
    border-color: #3b82f6 !important;
    color: #e2e8f0 !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 0 40px 0;
    color: #334155;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 16px; }
.empty-state-title { font-size: 1.3rem; font-weight: 600; color: #475569; margin-bottom: 8px; }
.empty-state-sub { font-size: 0.95rem; color: #334155; }

/* ── Result card ── */
.result-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: #475569; }
.result-question {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid #334155;
    line-height: 1.5;
}

/* ── Result meta row ── */
.meta-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #1e293b;
}
.meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.88rem;
    font-weight: 500;
    background: #0f172a;
    border: 1px solid #334155;
    color: #94a3b8;
}
.chip-green  { background: #052e16; border-color: #166534; color: #4ade80; }
.chip-blue   { background: #0c1a38; border-color: #1d4ed8; color: #60a5fa; }
.chip-yellow { background: #1c1500; border-color: #713f12; color: #fbbf24; }
.chip-gray   { background: #0f172a; border-color: #334155; color: #94a3b8; }

/* ── Trace ── */
.trace-sq {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 500;
    padding: 10px 0 8px 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 8px;
}
.trace-sq:last-child { border-bottom: none; }
.trace-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
.t-tag {
    font-size: 0.85rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 4px;
}
.t-hit  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.t-miss { background: #1c1500; color: #fbbf24; border: 1px solid #713f12; }
.t-web  { background: #0c1a38; color: #60a5fa; border: 1px solid #1d4ed8; }
.trace-queries { font-size: 0.85rem; color: #475569; margin-top: 4px; }

/* ── Sources ── */
.sources-wrap { display: flex; flex-wrap: wrap; gap: 8px; padding: 6px 0; }
.src-link {
    font-size: 0.85rem;
    color: #64748b;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 5px;
    padding: 5px 12px;
    text-decoration: none;
    word-break: break-all;
    transition: color 0.15s, border-color 0.15s;
}
.src-link:hover { color: #94a3b8; border-color: #334155; }

/* ── Sidebar ── */
.sb-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}
.sb-bignum {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.sb-hint { font-size: 0.85rem; color: #475569; margin-top: 4px; }

.mem-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 0;
    border-bottom: 1px solid #1e293b;
}
.mem-row:last-child { border-bottom: none; }
.mem-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.mem-label { font-size: 0.95rem; color: #64748b; flex: 1; }
.mem-val   { font-size: 0.95rem; color: #cbd5e1; font-weight: 600; }

.budget-bar-bg {
    background: #1e293b;
    border-radius: 4px;
    height: 7px;
    margin: 10px 0 5px 0;
    overflow: hidden;
}
.budget-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}

.hist-item {
    font-size: 0.9rem;
    color: #475569;
    padding: 8px 0;
    border-bottom: 1px solid #1e293b;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    cursor: default;
}
.hist-item:hover { color: #64748b; }
.hist-num { color: #334155; margin-right: 6px; font-size: 0.8rem; }

/* ── Sidebar buttons ── */
.stButton button {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    padding: 10px 14px !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.stButton button:hover { border-color: #475569 !important; color: #e2e8f0 !important; }

/* ── Expander ── */
details summary {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
}
details[open] summary { color: #94a3b8 !important; }

/* ── Metrics ── */
[data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}
[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px !important;
}

/* ── Global overrides ── */
.stMarkdown p, .stMarkdown li {
    font-size: 0.95rem !important;
    color: #cbd5e1 !important;
    line-height: 1.8 !important;
}
.stMarkdown li { margin-bottom: 4px !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    margin-top: 18px !important;
}
.stMarkdown strong { color: #e2e8f0 !important; font-weight: 600 !important; }
hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = ResearchAgent()
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

agent: ResearchAgent = st.session_state.agent

EXAMPLE_QUERIES = [
    "What are the economic impacts of AI on developing countries and what policies are being proposed?",
    "How does RAG compare to fine-tuning for enterprise LLM applications, and when should each be used?",
    "What caused the 2023 banking crisis and how does it compare to 2008?",
]

# ── Live stats ─────────────────────────────────────────────────────
budget        = agent.budget.summary()
wm            = agent.working.token_usage()
session_cap   = budget["session_budget_usd"] or SESSION_COST_LIMIT
cost_pct      = min(budget["session_cost_usd"] / session_cap, 1.0)
cost_color    = "#ef4444" if cost_pct > 0.8 else "#f59e0b" if cost_pct > 0.5 else "#4ade80"
total_tok     = budget["total_input_tokens"] + budget["total_output_tokens"]
searches_left = max(0, MAX_WEB_SEARCHES - agent.budget.web_searches_this_query)
wm_pct        = min(wm["used"] / wm["limit"], 1.0) if wm.get("limit") else 0
wm_color      = "#ef4444" if wm_pct > 0.8 else "#f59e0b" if wm_pct > 0.5 else "#818cf8"

if cost_pct >= 1.0:
    status_cls, status_label = "status-exceeded", "Budget Exceeded"
elif cost_pct > 0.8:
    status_cls, status_label = "status-warning", "Budget Warning"
else:
    status_cls, status_label = "status-ready", "Ready"

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">Session Budget</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:4px">
        <span class="sb-bignum" style="color:{cost_color}">${budget['session_cost_usd']:.4f}</span>
        <span style="font-size:0.85rem;color:#475569;margin-left:6px">/ ${session_cap:.2f}</span>
    </div>
    <div class="budget-bar-bg">
        <div class="budget-bar-fill" style="width:{cost_pct*100:.1f}%;background:{cost_color}"></div>
    </div>
    <div class="sb-hint">{cost_pct*100:.1f}% used &nbsp;·&nbsp; ${budget['budget_remaining_usd']:.4f} remaining</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Token Usage</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:4px">
        <span class="sb-bignum">{total_tok:,}</span>
        <span style="font-size:0.85rem;color:#475569;margin-left:6px">tokens</span>
    </div>
    <div class="sb-hint">Context limit: {budget['context_token_limit']:,} / call</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Memory Tiers</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="mem-row">
        <div class="mem-dot" style="background:{wm_color}"></div>
        <div class="mem-label">Tier 1 · Working</div>
        <div class="mem-val">{wm['used']} tok</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="budget-bar-bg" style="margin:0 0 12px 19px">
        <div class="budget-bar-fill" style="width:{wm_pct*100:.1f}%;background:{wm_color}"></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="mem-row">
        <div class="mem-dot" style="background:#38bdf8"></div>
        <div class="mem-label">Tier 2 · Episodic</div>
        <div class="mem-val">{agent.episodic.count()} chunks</div>
    </div>
    <div class="mem-row">
        <div class="mem-dot" style="background:#34d399"></div>
        <div class="mem-label">Tier 3 · Facts</div>
        <div class="mem-val">{agent.episodic.count_facts()} facts</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Save session to memory", use_container_width=True):
        try:
            summary = agent.end_session()
            st.success("Session saved.")
            st.caption(summary)
        except Exception as exc:
            st.error(str(exc))

    if st.button("New session", use_container_width=True):
        st.session_state.agent   = ResearchAgent()
        st.session_state.history = []
        st.session_state.prefill = ""
        st.rerun()

    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sb-title">History</div>', unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.history), 1):
            q = item["query"]
            st.markdown(
                f'<div class="hist-item"><span class="hist-num">{i}</span>{q[:52]}{"…" if len(q)>52 else ""}</div>',
                unsafe_allow_html=True,
            )

# ── Header ─────────────────────────────────────────────────────────
st.markdown(
    f'<div class="dash-title">🔍 Research Agent '
    f'<span class="status-badge {status_cls}">{status_label}</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="dash-sub">'
    f'Memory-first · <b>{CONTEXT_TOKEN_LIMIT // 1000}K token context limit</b> · '
    f'<b>${SESSION_COST_LIMIT:.2f} session budget</b> · Claude native tool use'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Budget / exceeded banners ───────────────────────────────────────
if cost_pct >= 1.0:
    st.markdown(
        '<div class="exceeded-banner">⛔ Session budget exceeded. Start a new session or save and reset.</div>',
        unsafe_allow_html=True,
    )
elif cost_pct > 0.8:
    st.markdown(
        f'<div class="warn-banner">⚠️ Budget at {cost_pct*100:.0f}% — consider saving this session before it runs out.</div>',
        unsafe_allow_html=True,
    )

# ── Live stats bar ─────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Session Cost",   f"${budget['session_cost_usd']:.4f}")
c2.metric("Tokens Used",    f"{total_tok:,}")
c3.metric("Memory Chunks",  agent.episodic.count())
c4.metric("Working Memory", f"{wm['used']} tok")
c5.metric("Searches Left",  searches_left)

st.markdown("<br>", unsafe_allow_html=True)

# ── Query form ─────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=True):
    st.markdown('<div class="query-label">Research question</div>', unsafe_allow_html=True)
    user_query = st.text_area(
        label="query",
        placeholder="Ask a complex, multi-part question…",
        height=95,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Search", use_container_width=True)

# ── Example queries (shown always, subtle when history exists) ──────
opacity = "0.45" if st.session_state.history else "1"
st.markdown(
    f'<div class="example-section-title" style="opacity:{opacity}">Try an example</div>',
    unsafe_allow_html=True,
)
ex_cols = st.columns(3)
for col, q in zip(ex_cols, EXAMPLE_QUERIES):
    with col:
        if st.button(q[:72] + ("…" if len(q) > 72 else ""), use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

# ── Run query (form submit or example button) ───────────────────────
query_to_run = user_query.strip() if submitted and user_query.strip() else st.session_state.pending_query

if query_to_run:
    st.session_state.pending_query = ""
    with st.spinner("Researching…"):
        try:
            result = agent.query(query_to_run)
        except Exception as exc:
            st.error(str(exc))
            result = None
    if result:
        st.session_state.history.append({"query": query_to_run, "result": result})
        st.rerun()

# ── Empty state ────────────────────────────────────────────────────
if not st.session_state.history:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🧠</div>
        <div class="empty-state-title">No queries yet</div>
        <div class="empty-state-sub">Ask a complex question above or click an example to get started.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Results ────────────────────────────────────────────────────────
for item in reversed(st.session_state.history):
    q = item["query"]
    r = item["result"]
    b = r["budget"]

    st.markdown(f"""
    <div class="result-card">
        <div class="result-question">{q}</div>
    """, unsafe_allow_html=True)

    st.markdown(r["answer"])

    mem_cls  = "chip-green"  if r["memory_hits"] > 0  else "chip-gray"
    web_cls  = "chip-blue"   if r["web_searches"] > 0 else "chip-gray"
    cost_cls = "chip-yellow" if b["session_cost_usd"] > 0.02 else "chip-gray"

    st.markdown(f"""
    <div class="meta-row">
        <span class="meta-chip {mem_cls}">● {r['memory_hits']} memory hits</span>
        <span class="meta-chip {web_cls}">↗ {r['web_searches']} web searches</span>
        <span class="meta-chip chip-gray">⚡ {r['tool_calls']} tool calls</span>
        <span class="meta-chip chip-gray">◈ {len(r['sources_used'])} sources</span>
        <span class="meta-chip {cost_cls}">${b['session_cost_usd']:.4f} spent</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How this was researched"):
        for step in r.get("trace", []):
            st.markdown(
                f'<div class="trace-sq">{step["sub_question"]}</div>',
                unsafe_allow_html=True,
            )
            if step["trace"]:
                tags, queries = "", []
                for t in step["trace"]:
                    if t["tool"] == "query_memory":
                        cls  = "t-hit"  if t["hit"] else "t-miss"
                        icon = "● memory hit" if t["hit"] else "○ memory miss"
                    else:
                        cls, icon = "t-web", "↗ web search"
                    tags += f'<span class="t-tag {cls}">{icon}</span>'
                    queries.append(t["query"][:60])
                st.markdown(
                    f'<div class="trace-tags">{tags}</div>'
                    f'<div class="trace-queries">{" · ".join(queries)}</div>',
                    unsafe_allow_html=True,
                )

    if r["sources_used"]:
        with st.expander(f"Sources  ({len(r['sources_used'])})"):
            links = "".join(
                f'<a href="{s}" target="_blank" class="src-link">{s}</a>'
                for s in r["sources_used"]
            )
            st.markdown(f'<div class="sources-wrap">{links}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

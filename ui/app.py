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
    page_title="Research Agent",
    page_icon="",
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
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.dash-sub {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 24px;
}
.dash-sub b { color: #94a3b8; font-weight: 500; }

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
    font-size: 1.1rem !important;
    line-height: 1.7 !important;
    padding: 14px 16px !important;
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
}
[data-testid="stFormSubmitButton"] button:hover { background: #2563eb !important; }

/* ── Result card ── */
.result-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.result-question {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 18px;
    padding-bottom: 16px;
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
.budget-bar-fill { height: 100%; border-radius: 4px; }

.hist-item {
    font-size: 0.9rem;
    color: #475569;
    padding: 8px 0;
    border-bottom: 1px solid #1e293b;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
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
}
.stButton button:hover { border-color: #475569 !important; color: #e2e8f0 !important; }

/* ── Expander ── */
details summary {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
}
details[open] summary { color: #94a3b8 !important; }

/* ── Global overrides ── */
.stMarkdown p, .stMarkdown li {
    font-size: 1.05rem !important;
    color: #cbd5e1 !important;
    line-height: 1.9 !important;
}
.stMarkdown li { margin-bottom: 6px !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    margin-top: 22px !important;
}
.stMarkdown strong { color: #e2e8f0 !important; font-weight: 600 !important; }
hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }
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
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = ResearchAgent()
if "history" not in st.session_state:
    st.session_state.history = []

agent: ResearchAgent = st.session_state.agent

# ── Live stats ─────────────────────────────────────────────────────
budget      = agent.budget.summary()
wm          = agent.working.token_usage()
session_cap = budget["session_budget_usd"] or 0.05
cost_pct    = min(budget["session_cost_usd"] / session_cap, 1.0)
cost_color  = "#ef4444" if cost_pct > 0.8 else "#f59e0b" if cost_pct > 0.5 else "#4ade80"
searches_left = MAX_WEB_SEARCHES - agent.budget.web_searches_this_query

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
    total_tok = budget['total_input_tokens'] + budget['total_output_tokens']
    st.markdown(f"""
    <div class="sb-bignum">{total_tok:,}</div>
    <div class="sb-hint">Context limit: {budget['context_token_limit']:,} / call</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Memory Tiers</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="mem-row">
        <div class="mem-dot" style="background:#818cf8"></div>
        <div class="mem-label">Tier 1 · Working</div>
        <div class="mem-val">{wm['used']} tok</div>
    </div>
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
            st.success("Saved.")
            st.caption(summary)
        except Exception as exc:
            st.error(str(exc))

    if st.button("New session", use_container_width=True):
        st.session_state.agent   = ResearchAgent()
        st.session_state.history = []
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

# ── Main ───────────────────────────────────────────────────────────
st.markdown('<div class="dash-title">Research Agent</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="dash-sub">'
    f'Memory-first · <b>{CONTEXT_TOKEN_LIMIT // 1000}K token context limit</b> · <b>${SESSION_COST_LIMIT:.2f} session budget</b> · Claude native tool use'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Live dashboard stats bar ───────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Session Cost",      f"${budget['session_cost_usd']:.4f}")
c2.metric("Tokens Used",       f"{total_tok:,}")
c3.metric("Memory Chunks",     agent.episodic.count())
c4.metric("Working Memory",    f"{wm['used']} tok")
c5.metric("Searches Left",     searches_left)

st.markdown("<br>", unsafe_allow_html=True)

# ── Query form ─────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=True):
    st.markdown('<div class="query-label">Research question</div>', unsafe_allow_html=True)
    user_query = st.text_area(
        label="query",
        placeholder="Ask a complex, multi-part question — e.g. What are the economic impacts of AI on developing countries and what policies are being proposed?",
        height=95,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Search", use_container_width=True)

if submitted and user_query.strip():
    with st.spinner("Researching…"):
        try:
            result = agent.query(user_query.strip())
        except Exception as exc:
            st.error(str(exc))
            result = None
    if result:
        st.session_state.history.append({"query": user_query.strip(), "result": result})
        st.rerun()

# ── Results ────────────────────────────────────────────────────────
for item in reversed(st.session_state.history):
    q = item["query"]
    r = item["result"]
    b = r["budget"]

    # Result card
    st.markdown(f"""
    <div class="result-card">
        <div class="result-question">{q}</div>
        <div class="result-body">
    """, unsafe_allow_html=True)

    st.markdown(r["answer"])

    # Meta chips
    mem_cls  = "chip-green"  if r["memory_hits"] > 0   else "chip-gray"
    web_cls  = "chip-blue"   if r["web_searches"] > 0  else "chip-gray"
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
    </div>
    """, unsafe_allow_html=True)

    # Reasoning trace
    with st.expander("How this was researched"):
        for step in r.get("trace", []):
            st.markdown(
                f'<div class="trace-sq">{step["sub_question"]}</div>',
                unsafe_allow_html=True,
            )
            if step["trace"]:
                tags = ""
                queries = []
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

    # Sources
    if r["sources_used"]:
        with st.expander(f"Sources  ({len(r['sources_used'])})"):
            links = "".join(
                f'<a href="{s}" target="_blank" class="src-link">{s}</a>'
                for s in r["sources_used"]
            )
            st.markdown(f'<div class="sources-wrap">{links}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

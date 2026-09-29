"""Agent Memory SDK — Streamlit Dashboard (demo / test UI).

Launch:
    agent-memory-dashboard
    AGENT_MEMORY_DIR=/path/to/data agent-memory-dashboard
"""
from __future__ import annotations

import os
from typing import Any

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "Install with: pip install agent-memory-sdk[dashboard]"
    ) from None

st.set_page_config(
    page_title="Agent Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agent_memory.manager import Memory  # noqa: E402
from agent_memory.models import MemoryScope, MemoryType  # noqa: E402

# ── Minimal CSS overrides ─────────────────────────────────────────────────────
st.markdown("""
<style>
.stMainBlockContainer { padding-top: 3.5rem !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — connection ──────────────────────────────────────────────────────
_DIR  = os.environ.get("AGENT_MEMORY_DIR", ".agent_memory")
_COL  = os.environ.get("AGENT_MEMORY_COLLECTION", "agent_memories")
_BACK = os.environ.get("AGENT_MEMORY_BACKEND", "sqlite")

with st.sidebar:
    st.markdown("## 🧠 Agent Memory")
    st.divider()

    data_dir   = st.text_input("Data directory", value=st.session_state.get("data_dir", _DIR))
    collection = st.text_input("Collection", value=st.session_state.get("collection", _COL))
    backend    = st.selectbox("Backend", ["sqlite", "chromadb", "redis", "postgres"],
                               index=["sqlite","chromadb","redis","postgres"].index(
                                   st.session_state.get("backend", _BACK)))

    c1, c2 = st.columns(2)
    if c1.button("Apply", use_container_width=True):
        st.session_state["data_dir"]   = data_dir
        st.session_state["collection"] = collection
        st.session_state["backend"]    = backend
        st.cache_resource.clear()
        st.rerun()
    if c2.button("🔄 Refresh", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.caption("**Seed demo data:**")
    st.code("python scripts/seed_demo.py\n  --data-dir PATH", language="bash")

# ── Memory singleton ──────────────────────────────────────────────────────────
@st.cache_resource
def _get_mem(d: str, c: str, b: str) -> Memory:  # type: ignore[return]
    return Memory(persist_dir=d, collection_name=c, backend=b)

def mem() -> Memory:
    return _get_mem(  # type: ignore[no-any-return]
        st.session_state.get("data_dir", _DIR),
        st.session_state.get("collection", _COL),
        st.session_state.get("backend", _BACK),
    )

try:
    _mem   = mem()
    _stats = _mem.stats()
except Exception as exc:
    st.error(f"**Cannot connect:** {exc}")
    st.info("Set the **Data directory** in the sidebar or run `python scripts/seed_demo.py`.")
    st.stop()

total: int        = _stats.get("total", 0)
by_state          = _stats.get("by_state", {})
by_type           = _stats.get("by_type", {})
total_access: int = _stats.get("total_access_count", 0)

# ── KPI strip — custom HTML tiles (st.metric label visibility is broken in newer
# Streamlit versions that set visibility:hidden via the visibility="0" attribute
# with higher-specificity stylesheet rules that !important cannot override reliably)
def _kpi(col: Any, label: str, value: int) -> None:
    col.markdown(
        f"""<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
        padding:12px 16px;margin-bottom:4px">
        <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;
        letter-spacing:.5px;font-weight:600;margin-bottom:4px">{label}</div>
        <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )

k1, k2, k3, k4, k5 = st.columns(5)
_kpi(k1, "Total",    total)
_kpi(k2, "Active",   by_state.get("active", 0))
_kpi(k3, "Archived", by_state.get("archived", 0))
_kpi(k4, "Expired",  by_state.get("expired", 0))
_kpi(k5, "Accesses", total_access)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t_stats, t_browse, t_resolve = st.tabs(["📊 Stats", "📋 Memories", "🔍 Resolve"])

# =============================================================================
# STATS
# =============================================================================
with t_stats:
    if total == 0:
        st.info("No memories yet. Run `python scripts/seed_demo.py --data-dir PATH` to populate.")
    else:
        col_s, col_t = st.columns(2)

        with col_s:
            st.subheader("By State")
            try:
                import altair as alt
                import pandas as pd

                df = pd.DataFrame([{"State": k, "Count": v} for k, v in by_state.items()])
                ch = (
                    alt.Chart(df)
                    .mark_arc(innerRadius=55, outerRadius=105)
                    .encode(
                        theta=alt.Theta("Count:Q"),
                        color=alt.Color(
                            "State:N",
                            scale=alt.Scale(
                                domain=["active", "archived", "expired", "deleted"],
                                range=["#6366f1", "#f59e0b", "#ef4444", "#6b7280"],
                            ),
                            legend=alt.Legend(orient="bottom"),
                        ),
                        tooltip=["State:N", "Count:Q"],
                    )
                    .properties(height=280)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(ch, use_container_width=True)
            except ImportError:
                for s, c in by_state.items():
                    st.progress(c / max(total, 1), text=f"{s}: {c}")

        with col_t:
            st.subheader("By Type")
            try:
                import altair as alt
                import pandas as pd

                df2 = pd.DataFrame([
                    {"Type": k, "Count": v}
                    for k, v in sorted(by_type.items(), key=lambda x: -x[1])
                ])
                ch2 = (
                    alt.Chart(df2)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#6366f1")
                    .encode(
                        x=alt.X("Type:N", sort="-y", axis=alt.Axis(labelAngle=-30, labelFontSize=11)),
                        y=alt.Y("Count:Q", axis=alt.Axis(tickMinStep=1)),
                        tooltip=["Type:N", "Count:Q"],
                    )
                    .properties(height=280)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(ch2, use_container_width=True)
            except ImportError:
                for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
                    st.progress(c / max(total, 1), text=f"{t}: {c}")

# =============================================================================
# MEMORIES
# =============================================================================
with t_browse:
    fc1, fc2, fc3 = st.columns([3, 2, 2])
    with fc1:
        q = st.text_input("Search", placeholder="keyword or phrase…",
                          label_visibility="collapsed", key="br_q")
    with fc2:
        sc = st.selectbox("Scope", ["all"] + [s.value for s in MemoryScope],
                          label_visibility="collapsed", key="br_sc")
    with fc3:
        ty = st.selectbox("Type", ["all"] + [t.value for t in MemoryType],
                          label_visibility="collapsed", key="br_ty")

    scopes   = None if sc == "all" else [sc]
    mem_type = None if ty == "all" else ty

    if q.strip():
        raw     = _mem.store.keyword_search(
            q, top_k=40,
            scopes=[MemoryScope(s) for s in scopes] if scopes else None,
        )
        entries = [e for e, _ in raw]
    else:
        entries = _mem.list(limit=40, scope=scopes, type=mem_type)

    st.caption(f"{len(entries)} memories shown")

    if not entries:
        st.info("No memories match.")
    else:
        try:
            import pandas as pd

            ICON = {"active": "🟢", "archived": "🟡", "expired": "🔴"}
            df_b = pd.DataFrame([{
                " ":        ICON.get(e.state.value, "⚪"),
                "Query":    e.query[:75],
                "Type":     e.type.value,
                "Scope":    e.scope.value,
                "Conf %":   int(e.confidence * 100),
                "Accessed": e.access_count,
                "Updated":  e.updated_at.strftime("%Y-%m-%d"),
            } for e in entries])
            st.dataframe(df_b, use_container_width=True, height=320, hide_index=True)
        except ImportError:
            for e in entries:
                st.text(f"[{e.type.value}] {e.query[:70]}")

    st.divider()
    with st.expander("➕ Add memory"):
        with st.form("add_mem", clear_on_submit=True):
            nq, nr = st.text_input("Query *"), st.text_area("Response *", height=60)
            na, nb = st.columns(2)
            nt = na.selectbox("Type", [t.value for t in MemoryType])
            ns = nb.selectbox("Scope", [s.value for s in MemoryScope])
            if st.form_submit_button("Store", use_container_width=True):
                if nq.strip() and nr.strip():
                    _mem.remember(nq, nr, type=nt, scope=ns)
                    st.success("Stored ✓")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.warning("Query and response are required.")

# =============================================================================
# RESOLVE
# =============================================================================
with t_resolve:
    rl, rr = st.columns([3, 1])
    with rl:
        rq = st.text_area("Query", placeholder="Ask something the agent might remember…",
                          height=90, key="rv_q")
    with rr:
        rmode  = st.selectbox("Mode", ["auto","replay","restore","verify"], key="rv_mode")
        rtopk  = st.slider("Top K", 1, 10, 3, key="rv_k")
        rscope = st.multiselect("Scope filter", [s.value for s in MemoryScope], key="rv_sc")

    if st.button("▶ Resolve", type="primary", key="rv_btn"):
        if not rq.strip():
            st.warning("Enter a query above.")
        else:
            with st.spinner("Resolving…"):
                d = _mem.resolve(rq, mode=rmode, top_k=rtopk, scope=rscope or None)

            BADGE = {"replay":"#22c55e","restore":"#3b82f6","verify":"#f59e0b","none":"#6b7280"}
            col = BADGE.get(d.action.value, "#6b7280")
            st.markdown(
                f'<span style="background:{col};color:#fff;padding:4px 16px;'
                f'border-radius:999px;font-weight:700;">'
                f'{d.action.value.upper()}</span>'
                f'&ensp;confidence <b>{d.confidence:.2f}</b>',
                unsafe_allow_html=True,
            )
            if d.reasons:
                st.caption("  ·  ".join(d.reasons))

            st.write("")
            if d.action.value == "replay" and d.memory:
                st.success(f"**Response:** {d.memory.response}")
                st.caption(f"Memory `{d.memory.id[:16]}…` · "
                           f"accessed {d.memory.access_count}×")
            elif d.action.value in ("restore", "verify") and d.context:
                for i, ctx in enumerate(d.context, 1):
                    with st.expander(f"#{i} {ctx.entry.query[:70]}", expanded=i == 1):
                        st.write(f"**Response:** {ctx.entry.response}")
                        st.caption(f"score {ctx.final_score:.3f} · "
                                   f"{ctx.entry.type.value} · {ctx.entry.scope.value}")
            elif d.action.value == "none":
                st.warning("No relevant memory found.")

            if d.scores:
                with st.expander("Score breakdown"):
                    for k, v in d.scores.items():
                        st.progress(float(v), text=f"{k}: {float(v):.3f}")

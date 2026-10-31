"""
pages/1_Dashboard.py — Application overview and statistics.
"""
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard — Document Intelligence", page_icon="📊", layout="wide")

from modules.database_manager import get_all_documents, get_recent_activity, get_statistics

st.markdown("## 📊 Dashboard")
st.markdown("Real-time overview of your document library and system activity.")
st.markdown("---")

# ── Metrics ──────────────────────────────────────────────────────────────────
stats = get_statistics()

col1, col2, col3, col4, col5, col6 = st.columns(6)
metrics = [
    ("📄", "Total Documents", stats.get("total_docs", 0)),
    ("✅", "Processed", stats.get("completed_docs", 0)),
    ("📃", "Pages", stats.get("total_pages", 0)),
    ("📊", "Tables", stats.get("total_tables", 0)),
    ("🖼️", "Images", stats.get("total_images", 0)),
    ("🔢", "Chunks", stats.get("total_chunks", 0)),
]
for col, (icon, label, value) in zip([col1, col2, col3, col4, col5, col6], metrics):
    with col:
        st.metric(label=f"{icon} {label}", value=f"{value:,}")

st.markdown("---")

# ── Document status chart ─────────────────────────────────────────────────────
docs = get_all_documents()

col_chart, col_recent = st.columns([1, 1])

with col_chart:
    st.markdown("### 📈 Document Status Distribution")
    if docs:
        status_counts: dict = {}
        for d in docs:
            s = d.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1

        colors = {
            "completed": "#3fb950",
            "processing": "#d29922",
            "failed": "#f85149",
            "pending": "#8b949e",
        }
        labels = list(status_counts.keys())
        values = list(status_counts.values())
        clrs = [colors.get(l, "#58a6ff") for l in labels]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=clrs),
                hole=0.5,
                textinfo="label+value",
            )
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            showlegend=False,
            margin=dict(t=20, b=20, l=0, r=0),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No documents uploaded yet.")

with col_recent:
    st.markdown("### 🕐 Recent Documents")
    recent_docs = docs[:8]
    if recent_docs:
        for d in recent_docs:
            status = d.get("status", "unknown")
            badge = {
                "completed": "🟢",
                "processing": "🟡",
                "failed": "🔴",
                "pending": "⚪",
            }.get(status, "⚪")
            ts = d.get("upload_ts", "")[:10]
            st.markdown(
                f"{badge} **{d['filename']}** &nbsp;&nbsp; "
                f"<span style='color:#8b949e;font-size:0.8rem'>{d['file_type'].upper()} · {ts}</span>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No documents yet. Upload one to get started.")

st.markdown("---")

# ── Processing stats bar chart ────────────────────────────────────────────────
if docs and any(d.get("page_count", 0) for d in docs):
    st.markdown("### 📋 Per-document Statistics")
    completed_docs = [d for d in docs if d.get("status") == "completed"][:10]
    if completed_docs:
        names = [d["filename"][:30] for d in completed_docs]
        pages = [d.get("page_count") or 0 for d in completed_docs]
        tables = [d.get("table_count") or 0 for d in completed_docs]
        images = [d.get("image_count") or 0 for d in completed_docs]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Pages", x=names, y=pages, marker_color="#58a6ff"))
        fig2.add_trace(go.Bar(name="Tables", x=names, y=tables, marker_color="#3fb950"))
        fig2.add_trace(go.Bar(name="Images", x=names, y=images, marker_color="#bc8cff"))
        fig2.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            xaxis=dict(tickangle=-20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=300,
            margin=dict(t=40, b=10, l=0, r=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Activity log ─────────────────────────────────────────────────────────────
st.markdown("### 📝 Recent Activity")
activity = get_recent_activity(limit=10)
if activity:
    for item in activity:
        ts = item.get("ts", "")[:19].replace("T", " ")
        action = item.get("action", "")
        details = item.get("details", "")
        st.caption(f"`{ts}` — **{action}**: {details}")
else:
    st.info("No activity yet.")

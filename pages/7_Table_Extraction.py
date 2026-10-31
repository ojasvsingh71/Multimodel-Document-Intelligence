"""
pages/7_Table_Extraction.py
Browse, view, and export tables extracted from documents.
"""
import streamlit as st

st.set_page_config(
    page_title="Table Extraction — Document Intelligence",
    page_icon="📊",
    layout="wide",
)

from modules.database_manager import get_all_documents
from modules.table_extractor import (
    load_tables_for_document,
    table_to_dataframe,
    table_to_csv_bytes,
    table_to_excel_bytes,
    table_to_markdown,
)

st.markdown("## 📊 Table Extraction")
st.markdown("View and export tables extracted from your documents.")
st.markdown("---")

docs = get_all_documents()
docs_with_tables = [d for d in docs if d.get("status") == "completed" and d.get("table_count", 0) > 0]

if not docs_with_tables:
    st.info(
        "No tables found. Upload a document containing tables. "
        "Tables are automatically extracted during document processing."
    )
    st.stop()

# ── Document selector ─────────────────────────────────────────────────────────
selected_filename = st.selectbox(
    "Select document",
    options=[d["filename"] for d in docs_with_tables],
)
selected_doc = next(d for d in docs_with_tables if d["filename"] == selected_filename)
doc_id = selected_doc["doc_id"]

tables = load_tables_for_document(doc_id)

if not tables:
    st.warning("No tables could be loaded from this document's processed data.")
    st.stop()

st.markdown(f"**{len(tables)} table(s) found in {selected_filename}**")
st.markdown("---")

# ── Table viewer ──────────────────────────────────────────────────────────────
for i, table in enumerate(tables):
    caption = table.get("caption", f"Table {i+1}")
    page = table.get("page")
    page_str = f" (Page {page})" if page else ""

    with st.expander(f"📋 {caption}{page_str}", expanded=(i == 0)):
        df = table_to_dataframe(table)

        if df.empty:
            # Fallback: show raw markdown
            md = table_to_markdown(table)
            if md and md != "_No data_":
                st.markdown(md)
            else:
                st.warning("Table appears to be empty or could not be parsed.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

        # Export buttons
        col1, col2, col3 = st.columns(3)
        safe_caption = caption.replace(" ", "_")[:30]

        with col1:
            csv_bytes = table_to_csv_bytes(table)
            st.download_button(
                "⬇️ Export CSV",
                data=csv_bytes,
                file_name=f"{safe_caption}.csv",
                mime="text/csv",
                key=f"csv_{i}_{doc_id}",
            )

        with col2:
            try:
                excel_bytes = table_to_excel_bytes(table)
                st.download_button(
                    "⬇️ Export Excel",
                    data=excel_bytes,
                    file_name=f"{safe_caption}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"excel_{i}_{doc_id}",
                )
            except Exception as e:
                st.caption(f"Excel export unavailable: {e}")

        with col3:
            md_content = table_to_markdown(table)
            st.download_button(
                "⬇️ Export Markdown",
                data=md_content,
                file_name=f"{safe_caption}.md",
                mime="text/markdown",
                key=f"md_{i}_{doc_id}",
            )

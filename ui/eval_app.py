# ui/eval_app.py

"""
Streamlit Evaluation Dashboard

Used in Phase 6 (RAGAS evaluation).
Shows evaluation metrics, test results, and quality scores.
Placeholder for now — fully built in Phase 6.
"""

import streamlit as st

st.set_page_config(
    page_title="RAG Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 RAG Evaluation Dashboard")
st.markdown("---")

st.info(
    "🔧 This dashboard is built in Phase 6 — RAGAS Evaluation Suite.\n\n"
    "It will show:\n"
    "- Faithfulness scores\n"
    "- Answer relevancy metrics\n"
    "- Context precision & recall\n"
    "- Golden dataset test results\n"
    "- Judge LLM evaluations"
)

st.markdown("### Coming in Phase 6:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Faithfulness", "—", help="How grounded is the answer in context")
with col2:
    st.metric("Relevancy", "—", help="How relevant is the answer to the query")
with col3:
    st.metric("Precision", "—", help="Context precision score")
with col4:
    st.metric("Recall", "—", help="Context recall score")

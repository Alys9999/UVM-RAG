"""
Streamlit front end for the UVM / SystemVerilog dense RAG system.

- Uses backend.run_rag_query() as RAG engine.
- Displays answer with [i] citations.
- Shows retrieved context with clickable uri+anchor links.
"""

import os
from typing import List

import streamlit as st
from haystack import Document  # type: ignore

from backend import run_rag_query


# ---------------------------------------------------------------------
# 1. Basic page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="UVM / SystemVerilog RAG",
    page_icon="📚",
    layout="wide",
)

st.title("UVM / SystemVerilog Dense RAG (GPU, Local Qwen)")
st.caption(
    "Dense-only, GPU-only Retrieval-Augmented Generation over UVM/SystemVerilog specs and code. "
    "All retrieval and LLM inference run locally."
)

# ---------------------------------------------------------------------
# 2. Sidebar options
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Search parameters")

    default_retriever_top_k = int(os.environ.get("UVM_RAG_RETRIEVER_TOP_K", "20"))
    default_answer_top_k = int(os.environ.get("UVM_RAG_ANSWER_TOP_K", "8"))

    retriever_top_k = st.slider(
        "Retriever top_k (PGVector)",
        min_value=1,
        max_value=20,
        value=default_retriever_top_k,
        step=1,
    )

    answer_top_k = st.slider(
        "Context docs in answer (N)",
        min_value=1,
        max_value=8,
        value=default_answer_top_k,
        step=1,
    )

    show_debug = st.checkbox("Show debug pipeline output", value=False)

    st.markdown("---")
    st.markdown(
        "Example questions:\n"
        "- *How does `uvm_packer` unpack dynamic arrays in UVM 1.2?*\n"
        "- *What is the purpose of `uvm_phase` and the run phases?*\n"
        "- *How do I implement a UVM sequence with `uvm_config_db`?*"
    )


# ---------------------------------------------------------------------
# 3. Helpers to render retrieved documents and citation mapping
# ---------------------------------------------------------------------

def render_context_docs(docs: List[Document]) -> None:
    if not docs:
        st.info("No context documents returned.")
        return

    for idx, d in enumerate(docs, start=1):
        section_title = d.meta.get("section_title", "Unknown section")
        doc_type = d.meta.get("type", "text")

        with st.expander(f"[{idx}] {section_title} ({doc_type})"):
            std = d.meta.get("std", "UVM-1.2")
            uri = d.meta.get("uri", "")
            anchor = d.meta.get("anchor", "")
            location = f"{uri}{anchor}" if uri else ""

            st.markdown(f"- **Standard:** `{std}`")
            st.markdown(f"- **Type:** `{doc_type}`")

            if location:
                st.markdown(f"- **Source:** [{location}]({location})")

            st.markdown("---")

            snippet = (d.content or "").strip()
            if len(snippet) > 1500:
                snippet = snippet[:1500] + " …"
            # We just show raw text; Streamlit syntax highlighting is not essential here.
            st.code(snippet, language="markdown")


def render_citation_table(docs: List[Document]) -> None:
    if not docs:
        return

    st.subheader("Citation mapping")

    for idx, d in enumerate(docs, start=1):
        section_title = d.meta.get("section_title", "Unknown section")
        doc_type = d.meta.get("type", "text")
        uri = d.meta.get("uri", "")
        anchor = d.meta.get("anchor", "")
        location = f"{uri}{anchor}" if uri else ""
        link_text = f"[{location}]({location})" if location else "_N/A_"

        st.markdown(
            f"- **[{idx}]** · `{doc_type}` · *{section_title}* · {link_text}",
            unsafe_allow_html=False,
        )


# Optional: turn [i] in answer into links when we have a corresponding doc with uri+anchor.
def linkify_citations(answer: str, docs: List[Document]) -> str:
    text = answer
    for idx, d in enumerate(docs, start=1):
        uri = d.meta.get("uri", "")
        anchor = d.meta.get("anchor", "")
        if not uri:
            continue
        target = f"{uri}{anchor}"
        marker = f"[{idx}]"
        link_marker = f"[{idx}]({target})"
        # Simple replacement; if the model repeats [i] many times, they all become links.
        text = text.replace(marker, link_marker)
    return text


# ---------------------------------------------------------------------
# 4. Main query area
# ---------------------------------------------------------------------

st.markdown("### Query")

default_question = "How does `uvm_packer` handle unpacking of arrays in UVM 1.2?"
query = st.text_area(
    label="Enter your question about UVM / SystemVerilog:",
    value=default_question,
    height=100,
)

col_run, col_clear = st.columns([1, 1])
with col_run:
    run_button = st.button("Search and answer", type="primary")
with col_clear:
    clear_button = st.button("Clear")

if clear_button:
    # Reset the page state
    st.rerun()

if run_button and query.strip():
    with st.spinner("Running dense retrieval and local HF generation…"):
        try:
            answer_text, docs_used, raw_result = run_rag_query(
                query=query.strip(),
                retriever_top_k=retriever_top_k,
                answer_top_k=answer_top_k,
            )
        except SystemExit as e:
            # Backend uses SystemExit if GPU is not available, etc.
            st.error(f"Backend aborted: {e}")
            print("there must be a GPU to run the chatbot locally!")
            st.stop()
        except Exception as e:
            st.exception(e)
            st.stop()

    col_answer, col_context = st.columns([2, 1])

    with col_answer:
        st.markdown("### Answer")

        if answer_text.strip():
            linked_answer = linkify_citations(answer_text, docs_used)
            st.markdown(linked_answer)
        else:
            st.info("No answer produced by the model.")

        render_citation_table(docs_used)

        if show_debug:
            st.markdown("---")
            st.markdown("#### Debug: raw pipeline output")
            st.json(raw_result)

    with col_context:
        st.markdown("### Retrieved context")
        render_context_docs(docs_used)

elif run_button and not query.strip():
    st.warning("Please enter a non-empty question.")

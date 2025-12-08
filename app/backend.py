"""
Backend RAG pipeline for UVM / SystemVerilog.

- GPU-only (fails hard if no CUDA).
- Dense-only retrieval (PgvectorEmbeddingRetriever).
- Local Hugging Face LLM via HuggingFaceLocalGenerator.
- Classical RAG: embed → retrieve → prompt → generate (no agents).

Usage from Streamlit (same process):

    import backend

    # 1) Once, when user clicks "Load models" or at startup:
    backend.load_rag_pipeline()

    # 2) For each query:
    answer, docs, raw = backend.run_rag_query("your question here")

This keeps model loading separate from querying and avoids re-loading on each call.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
import torch

from haystack import Pipeline, Document
from haystack.utils import ComponentDevice, Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever


# ---------------------------------------------------------------------
# 0. Enforce GPU-only execution
# ---------------------------------------------------------------------

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA GPU is required; CPU fallback is disabled.")

DEVICE = ComponentDevice.from_str("cuda:0")


# ---------------------------------------------------------------------
# 1. Configuration (via environment variables, with sane defaults)
# ---------------------------------------------------------------------

PG_TABLE_NAME = os.environ.get("UVM_RAG_PG_TABLE_NAME", "uvm_vert_docs")

# Must match the model + dimension used in step 03 (SentenceTransformersDocumentEmbedder).
EMBED_MODEL_NAME = os.environ.get("UVM_RAG_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DIM = int(os.environ.get("UVM_RAG_EMBED_DIM", "1024"))

# Retrieval defaults
RETRIEVER_TOP_K = int(os.environ.get("UVM_RAG_RETRIEVER_TOP_K", "20"))
ANSWER_TOP_K = int(os.environ.get("UVM_RAG_ANSWER_TOP_K", "8"))

# Local HF model for generation
HF_LOCAL_MODEL = os.environ.get("UVM_RAG_HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TASK = os.environ.get("UVM_RAG_HF_TASK", "text-generation")

# ---------------------------------------------------------------------
# 2. Connect to PgvectorDocumentStore (read-only in this stage)
# ---------------------------------------------------------------------

load_dotenv()
print("PG_CONN_STR =", os.getenv("PG_CONN_STR"))

document_store = PgvectorDocumentStore(
    connection_string=Secret.from_env_var("PG_CONN_STR"),
    table_name=PG_TABLE_NAME,
    embedding_dimension=EMBED_DIM,
    create_extension=True,
    vector_function="cosine_similarity",
    recreate_table=False,      # do NOT drop, index is already built in 03
    search_strategy="hnsw",    # or "exact" if you want brute-force
    keyword_index_name=f"haystack_keyword_index_{PG_TABLE_NAME}",
)

# ---------------------------------------------------------------------
# 3. Lazy-initialized components (global handles)
# ---------------------------------------------------------------------

_rag_pipeline: Optional[Pipeline] = None
_text_embedder: Optional[SentenceTransformersTextEmbedder] = None
_retriever: Optional[PgvectorEmbeddingRetriever] = None
_ranker: Optional[SentenceTransformersSimilarityRanker] = None
_prompt_builder: Optional[PromptBuilder] = None
_llm: Optional[HuggingFaceLocalGenerator] = None


# ---------------------------------------------------------------------
# 4. Prompt template
# ---------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """
You are an expert verification engineer specializing in SystemVerilog UVM.
Use ONLY the context documents below to answer the question.
If the answer is not clearly contained in the context, say that you do not know.

Question:
{{ query }}

Context documents:
{% for doc in documents %}
[{{ loop.index }}]
- std: {{ doc.meta.std | default("UVM-1.2") }}
- section_title: {{ doc.meta.section_title | default("N/A") }}
- type: {{ doc.meta.type | default("text") }}
- location: {{ doc.meta.uri | default("") }}{{ doc.meta.anchor | default("") }}

{{ doc.content }}

{% endfor %}

Instructions for answering:
- Answer concisely but precisely.
- Use correct UVM/SystemVerilog terminology.
- When you rely on a document, cite it as [index], for example [1], [2].
- If multiple documents support the same statement, you can cite like [1][3].

Answer:
"""


# ---------------------------------------------------------------------
# 5. Initialization entrypoint: load models and construct pipeline
# ---------------------------------------------------------------------

def load_rag_pipeline(force_reload: bool = False) -> Pipeline:
    """
    Initialize all heavy components (embedders, retriever, ranker, LLM)
    and build the RAG pipeline.

    - If the pipeline is already loaded and force_reload is False,
      this is a no-op and returns the existing pipeline.
    - If force_reload is True, the pipeline and models are re-initialized.

    This function is intended to be called once from the Streamlit app,
    before any calls to run_rag_query().
    """
    global _rag_pipeline, _text_embedder, _retriever, _ranker, _prompt_builder, _llm

    if _rag_pipeline is not None and not force_reload:
        print("[INFO] RAG pipeline already initialized; reusing existing instance.")
        return _rag_pipeline

    print("[INFO] Initializing RAG pipeline...")

    # 5.1 GPU-based query embedder
    _text_embedder = SentenceTransformersTextEmbedder(
        model=EMBED_MODEL_NAME,
        device=DEVICE,
    )
    _text_embedder.warm_up()
    print("[INFO] SentenceTransformersTextEmbedder warmed up on GPU.")

    # 5.2 Dense retriever + ranker
    _retriever = PgvectorEmbeddingRetriever(
        document_store=document_store,
        top_k=RETRIEVER_TOP_K,
    )

    _ranker = SentenceTransformersSimilarityRanker()
    _ranker.warm_up()
    print("[INFO] SentenceTransformersSimilarityRanker warmed up.")

    # 5.3 Prompt builder
    _prompt_builder = PromptBuilder(
        template=RAG_PROMPT_TEMPLATE,
        required_variables=["query"],
    )

    # 5.4 Local Hugging Face generator on GPU
    generation_kwargs = {
        "max_new_tokens": 512,
    }
    _llm = HuggingFaceLocalGenerator(
        model=HF_LOCAL_MODEL,
        task=HF_TASK,
        device=DEVICE,
        generation_kwargs=generation_kwargs,
    )
    _llm.warm_up()
    print("[INFO] HuggingFaceLocalGenerator warmed up with model:", HF_LOCAL_MODEL)

    # 5.5 Build pipeline graph
    pipe = Pipeline()
    pipe.add_component("text_embedder", _text_embedder)
    pipe.add_component("retriever", _retriever)
    pipe.add_component(instance=_ranker, name="ranker")
    pipe.add_component("prompt_builder", _prompt_builder)
    pipe.add_component("llm", _llm)

    # Connect query embedding → retriever
    pipe.connect("text_embedder", "retriever")
    pipe.connect("retriever", "ranker.documents")
    # Connect retrieved docs → prompt builder
    pipe.connect("ranker", "prompt_builder.documents")
    # Connect prompt → LLM
    pipe.connect("prompt_builder", "llm")

    print("[INFO] RAG pipeline topology constructed.")
    try:
        pipe.draw(path=Path("04_rag_query_hf_local_pipeline.png"))
        print(" - Saved pipeline graph to 04_rag_query_hf_local_pipeline.png")
    except Exception as e:
        print(" - Could not draw pipeline graph:", e)

    _rag_pipeline = pipe
    return _rag_pipeline


def is_rag_pipeline_loaded() -> bool:
    """
    Small helper to let the frontend check if the pipeline is ready.
    """
    return _rag_pipeline is not None


# ---------------------------------------------------------------------
# 6. Query entrypoint for UI/backend integration
# ---------------------------------------------------------------------

def run_rag_query(
    query: str,
    retriever_top_k: int | None = None,
    answer_top_k: int | None = None,
) -> Tuple[str, List[Document], Dict[str, Any]]:
    """
    Run the full RAG pipeline for a single query.

    - Requires that load_rag_pipeline() has been called first.
    - Embeds query on GPU (SentenceTransformersTextEmbedder).
    - Retrieves dense-only neighbors from PGVectorEmbeddingRetriever.
    - Reranks with SentenceTransformersSimilarityRanker.
    - Builds a citation-aware prompt (PromptBuilder).
    - Generates answer via HuggingFaceLocalGenerator on GPU.

    Returns:
        answer_text,
        docs_for_answer (top-N documents for UI),
        raw_result (full pipeline output for debugging).
    """
    if _rag_pipeline is None:
        raise RuntimeError(
            "RAG pipeline is not initialized. "
            "Call load_rag_pipeline() once before run_rag_query()."
        )

    if retriever_top_k is None:
        retriever_top_k = RETRIEVER_TOP_K
    if answer_top_k is None:
        answer_top_k = ANSWER_TOP_K

    result = _rag_pipeline.run(
        {
            "text_embedder": {"text": query},
            "retriever": {"top_k": retriever_top_k},
            "ranker": {"query": query, "top_k": answer_top_k},
            "prompt_builder": {"query": query},
        },
        include_outputs_from={"retriever", "prompt_builder", "llm"},
    )

    retrieved_docs: List[Document] = result["retriever"]["documents"]
    docs_for_answer = retrieved_docs[:answer_top_k]

    replies: List[str] = result["llm"]["replies"]
    answer_text = replies[0] if replies else ""

    return answer_text, docs_for_answer, result


# ---------------------------------------------------------------------
# Optional: auto-warm pipeline at import time when env is set
# ---------------------------------------------------------------------
AUTO_WARM_ENV = "UVM_RAG_AUTOWARM"

if os.getenv(AUTO_WARM_ENV, "0") == "1":
    # When UVM_RAG_AUTOWARM=1, we load the RAG pipeline as soon as
    # backend.py is imported (for example when the server or Streamlit
    # starts inside the Docker container).
    try:
        print(f"[INFO] {AUTO_WARM_ENV}=1 detected; pre-loading RAG pipeline...")
        load_rag_pipeline(force_reload=False)
        print("[INFO] RAG pipeline pre-loaded at import time.")
    except Exception as e:
        print("[ERROR] Failed to auto-warm RAG pipeline:", e)
        # Optional: re-raise if you want container startup to fail hard
        # raise


if __name__ == "__main__":
    # Example manual test from command line.
    test_query = "In UVM 1.2, what does uvm_packer::unpack_array do when unpacking dynamic arrays?"

    load_rag_pipeline()  # explicit one-time init

    answer_text, docs_used, raw = run_rag_query(
        query=test_query,
        retriever_top_k=15,
        answer_top_k=8,
    )

    print("=== Question ===")
    print(test_query)
    print("\n=== Answer (with [i] citations) ===")
    print(answer_text)

    print("\n=== Context documents (top-N used) ===")
    print("number of docs used:", len(docs_used))
    for idx, d in enumerate(docs_used, start=1):
        uri = d.meta.get("uri", "")
        anchor = d.meta.get("anchor", "")
        section_title = d.meta.get("section_title", "N/A")
        doc_type = d.meta.get("type", "text")
        print(f"\n[{idx}] ({doc_type}) {section_title}")
        print(f"    {uri}{anchor}")
        snippet = (d.content or "").strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        print(f"    {snippet}")

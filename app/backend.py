"""
Backend RAG pipeline for UVM / SystemVerilog:

- GPU-only (fails hard if no CUDA).
- Dense-only retrieval (PgvectorEmbeddingRetriever).
- Local Hugging Face LLM via HuggingFaceLocalGenerator.
- Classical RAG: embed → retrieve → prompt → generate (no agents).
"""

import os
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from pathlib import Path


import torch

from haystack import Pipeline, Document
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever

from haystack.utils import Secret


# ---------------------------------------------------------------------
# 0. Enforce GPU-only execution
# ---------------------------------------------------------------------

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA GPU is required; CPU fallback is disabled.")

DEVICE = ComponentDevice.from_str("cuda:0")

# ---------------------------------------------------------------------
# 1. Configuration (via environment variables, with sane defaults)
# ---------------------------------------------------------------------

# Connection string: prefer UVM_RAG_PG_CONN_STR, fall back to PG_CONN_STR, then default.
# PG_CONN_STR = os.environ.get(
#     "PG_CONN_STR",
#     "postgresql://postgres:postgres@localhost:5432/postgres",
# )

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
    connection_string = Secret.from_env_var("PG_CONN_STR"),
    table_name=PG_TABLE_NAME,
    embedding_dimension=EMBED_DIM,
    create_extension=True,
    
    vector_function="cosine_similarity",
    recreate_table=False,      # do NOT drop, index is already built in 03
    search_strategy="hnsw",    # or "exact" if you want brute-force
    keyword_index_name=f"haystack_keyword_index_{PG_TABLE_NAME}",
)

# ---------------------------------------------------------------------
# 3. Components: GPU text embedder, dense retriever, prompt builder, local HF LLM
# ---------------------------------------------------------------------

# 3.1 GPU-based query embedder
text_embedder = SentenceTransformersTextEmbedder(
    model=EMBED_MODEL_NAME,
    device=DEVICE,
)

text_embedder.warm_up()
print("[INFO] SentenceTransformersTextEmbedder warmed up on GPU.")

# 3.2 Dense-only PGVector retriever
retriever = PgvectorEmbeddingRetriever(
    document_store=document_store,
    top_k=RETRIEVER_TOP_K,
)

# 3.3 Citation-aware prompt template (non-chat, plain text)
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

prompt_builder = PromptBuilder(template=RAG_PROMPT_TEMPLATE)

# 3.4 Local Hugging Face generator on GPU
# You may want to tune generation_kwargs for deterministic / concise answers.
generation_kwargs = {
    "max_new_tokens": 512
}

hf_generator = HuggingFaceLocalGenerator(
    model=HF_LOCAL_MODEL,
    task=HF_TASK,
    device=DEVICE,
    generation_kwargs=generation_kwargs,
)

hf_generator.warm_up()
print("[INFO] HuggingFaceLocalGenerator warmed up with model:", HF_LOCAL_MODEL)


# ---------------------------------------------------------------------
# 4. Build Haystack v2 Pipeline graph
# ---------------------------------------------------------------------

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", hf_generator)

# Connect query embedding → retriever
rag_pipeline.connect("text_embedder", "retriever")

# Connect retrieved docs → prompt builder
rag_pipeline.connect("retriever", "prompt_builder.documents")

# Connect prompt → LLM
rag_pipeline.connect("prompt_builder", "llm")

print("[INFO] RAG pipeline topology constructed:")
#render a graph:
try:
    rag_pipeline.draw(path = Path("04_rag_query_hf_local_pipeline.png"))
    print(" - Saved pipeline graph to 04_rag_query_hf_local_pipeline.png")
except Exception as e:
    print(" - Could not draw pipeline graph:", e)

# ---------------------------------------------------------------------
# 5. Convenience function for UI/backend integration
# ---------------------------------------------------------------------

def run_rag_query(
    query: str,
    retriever_top_k: int | None = None,
    answer_top_k: int | None = None,
) -> Tuple[str, List[Document], Dict[str, Any]]:
    """
    Run the full RAG pipeline for a single query.

    - Embeds query on GPU (SentenceTransformersTextEmbedder).
    - Retrieves dense-only neighbors from PGVectorEmbeddingRetriever.
    - Builds a citation-aware prompt (PromptBuilder).
    - Generates answer via HuggingFaceLocalGenerator on GPU.
    - Returns:
        answer_text,
        docs_for_answer (top-N documents for UI),
        raw_result (full pipeline output for debugging).
    """
    if retriever_top_k is None:
        retriever_top_k = RETRIEVER_TOP_K
    if answer_top_k is None:
        answer_top_k = ANSWER_TOP_K

    # Run the whole graph once
    result = rag_pipeline.run(
        {
            "text_embedder": {"text": query},
            "retriever": {"top_k": retriever_top_k},
            "prompt_builder": {"query": query},
        },
        include_outputs_from={"retriever", "prompt_builder", "llm"},
    )

    retrieved_docs: List[Document] = result["retriever"]["documents"]
    # For display and prompt length control, we keep only the first answer_top_k
    docs_for_answer = retrieved_docs[:answer_top_k]

    replies: List[str] = result["llm"]["replies"]
    answer_text = replies[0] if replies else ""

    return answer_text, docs_for_answer, result

# main
if __name__ == "__main__":
    test_query = "In UVM 1.2, what does uvm_packer::unpack_array do when unpacking dynamic arrays?"

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
    print("number of docs used: " + str(len(docs_used)))
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


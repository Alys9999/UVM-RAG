# UVM-RAG

GPU-only, dense Retrieval-Augmented Generation (RAG) over SystemVerilog / UVM
documentation and related datasets, with a local Hugging Face LLM and a
Streamlit UI. The system lets you ask questions about UVM 1.2 and related
material and get citation-backed answers grounded in the official specs.

At a high level:

- Documents (UVM manuals, VERT data, supplemental datasets) are parsed and
  chunked via notebooks in `ingest/`.
- Chunks are embedded and stored in PostgreSQL with the `pgvector` extension.
- At query time, a Haystack pipeline retrieves the most relevant chunks,
  builds a prompt, and calls a local GPU LLM (Qwen) to generate an answer.
- A Streamlit app (`app/ui_streamlit.py`) provides the UI, tuning sliders, and
  document citations.

> Note: This project is explicitly **GPU-only**. The backend will terminate if
> no CUDA device is available.

---

## Repository layout

Top-level overview:

- `app/`
  - `backend.py` – Haystack RAG pipeline:
    - Enforces GPU-only execution (`torch.cuda.is_available()`).
    - Connects to PostgreSQL/pgvector via `PgvectorDocumentStore`.
    - Uses `SentenceTransformersTextEmbedder` for query embeddings.
    - Uses `PgvectorEmbeddingRetriever` + `SentenceTransformersSimilarityRanker`.
    - Builds prompts via `PromptBuilder` with a UVM-specific template.
    - Generates answers with `HuggingFaceLocalGenerator` (Qwen).
    - Exposes:
      - `load_rag_pipeline(force_reload: bool = False)` – lazy, one-time init.
      - `run_rag_query(query, retriever_top_k=None, answer_top_k=None)` – runs
        retrieval + generation and returns `(answer_text, docs_for_answer, raw)`.
      - `is_rag_pipeline_loaded()` – simple readiness check.
  - `ui_streamlit.py` – Streamlit front end:
    - Sidebar sliders for `retriever_top_k` and `answer_top_k`.
    - Text area for question input and a “Search and answer” button.
    - Renders answer with `[i]` citations; turns them into links when possible.
    - Shows a “Citation mapping” table and expandable context snippets.
    - Optional debug view to inspect the raw Haystack pipeline output.

- `ingest/`
  - Jupyter notebooks that implement the offline ingestion pipeline, in
    approximate order:
    - `01_toc_and_mineru_probe.ipynb` – probe document structure, TOC, and
      MinerU parsing behavior.
    - `01_1_postprocess_minerU.ipynb` – clean / post-process MinerU output.
    - `02_chunk_text.ipynb` – chunk cleaned text into retrieval-ready pieces.
    - `03_build_pgvector_index.ipynb` – create / configure the pgvector table
      and index in PostgreSQL.
    - `03_1_embed_and_store.ipynb` – embed chunks and write vectors + metadata
      into pgvector.
    - `04_rag_beckend.ipynb` – early RAG backend prototyping (now reflected in
      `app/backend.py`).

- `data/`
  - `UVM_Class_Reference_Manual_1.2.pdf`
  - `uvm_users_guide_1.2.pdf`
  - `VERT/`
    - `VERT.json` – UVM/VIP-related evaluation or training data.
  - `Supplimental_datasets/`
    - `VERT_withRAG.json`, `*dataset*.json` – additional case/if datasets used
      in experiments or evaluation.

- `work/`
  - Working directory for ingestion outputs:
    - `json_out/`, `VERT/`, `Supplimental_datasets/`,
      `work_class_reference/`, `work_manual/`.

- `uvm-rag/`
  - Local Python virtual environment (created on your machine). This is not
    required inside the Docker container.

- `uvm_db_dump.sql`
  - PostgreSQL dump (pg_dump) that:
    - Enables the `vector` extension.
    - Creates a `public.uvm_rag_docs` table with a 768-dim `vector` column,
      plus `content` and `meta` fields.
  - Restoring this dump gives you a ready-to-query database so you can skip
    re-running the ingestion notebooks if desired.

- `dockerfile`
  - GPU Docker image based on `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`.
  - Builds and installs Python 3.13.5 from source.
  - Installs a CUDA 11.8–compatible PyTorch stack:
    - `torch==2.7.1`, `torchvision==0.22.1`, `torchaudio==2.7.1`.
  - Copies the repo into `/app` and sets:
    - `PG_CONN_STR="postgresql://postgres:postgres@uvm-postgres-cloud:5432/postgres"`
    - `UVM_RAG_AUTOWARM=1` (auto-loads the RAG pipeline at startup).
    - `STREAMLIT_SERVER_ADDRESS`, `STREAMLIT_SERVER_PORT` for the UI.
  - Entrypoint is `docker_entry.sh`.

- `docker_entry.sh`
  - Installs Python dependencies from `requirements.txt` using Python 3.13.
  - Changes into `/app/app` and imports `backend` (which auto-warms the RAG
    pipeline because `UVM_RAG_AUTOWARM=1`).
  - Starts the Streamlit UI:
    - `streamlit run ui_streamlit.py --server.address=0.0.0.0 --server.port=8501`

- `requirements.txt`
  - Pinned dependencies for ingestion and the RAG system:
    - `haystack-ai`, `pgvector-haystack`, `pgvector`, `psycopg` / `psycopg2`.
    - `sentence-transformers`, `transformers`, Qwen-related utils.
    - `streamlit`, `mineru`, `doclayout_yolo`, PDF tooling, etc.

- `.env`
  - For local configuration of `PG_CONN_STR` and other env vars (not committed).

- `.dockerignore`, `.gitignore`
  - Standard ignore files.

---

## RAG architecture

The core RAG pipeline is defined in `app/backend.py` and built with Haystack:

1. **Embeddings**
   - `SentenceTransformersTextEmbedder` on GPU with model:
     - Default: `Qwen/Qwen3-Embedding-0.6B` (configurable).
   - Embedding dimension:
     - Default: `EMBED_DIM=1024` (configurable).

2. **Vector store (pgvector)**
   - `PgvectorDocumentStore` pointing at a PostgreSQL instance with the
     `vector` extension enabled.
   - Uses HNSW (`search_strategy="hnsw"`) by default.
   - Default table name:
     - `UVM_RAG_PG_TABLE_NAME` (default: `"uvm_vert_docs"`).
   - In `uvm_db_dump.sql`, the main table is `public.uvm_rag_docs` with
     `vector(768)`, so adjust your env vars to match whichever table/model you
     actually use.

3. **Retrieval + ranking**
   - `PgvectorEmbeddingRetriever` for dense-only retrieval.
   - `SentenceTransformersSimilarityRanker` to re-rank retrieved documents and
     select the top-N for context.

4. **Prompting**
   - `PromptBuilder` with `RAG_PROMPT_TEMPLATE`, which:
     - Treats the assistant as an expert verification engineer.
     - Prints document meta fields such as:
       - `std` (UVM standard), `section_title`, `type`, `uri`, `anchor`.
     - Instructs the model to:
       - Use only the provided context.
       - Answer concisely and precisely.
       - Cite sources as `[i]` (e.g., `[1]`, `[2][3]`).

5. **Generation**
   - `HuggingFaceLocalGenerator` on GPU with:
     - Default model: `Qwen/Qwen2.5-1.5B-Instruct`.
     - Default task: `"text-generation"`.
     - `max_new_tokens=512` (configurable).

6. **Query entrypoint**
   - `run_rag_query(query, retriever_top_k=None, answer_top_k=None)`:
     - Embeds the query.
     - Retrieves `retriever_top_k` neighbors from pgvector.
     - Re-ranks and keeps `answer_top_k` documents for context.
     - Builds a prompt and calls the local LLM.
     - Returns:
       - `answer_text` (string).
       - `docs_for_answer` (list of Haystack `Document` objects).
       - `raw_result` (full pipeline output, useful for debugging).

7. **GPU requirement**
   - At import time, `backend.py` checks `torch.cuda.is_available()` and calls
     `SystemExit` if no GPU is present.
   - The Streamlit UI catches this and shows a user-friendly error.

---

## Environment configuration

Key environment variables (with their defaults in `backend.py` and `dockerfile`):

- Database:
  - `PG_CONN_STR` – PostgreSQL connection string with pgvector enabled.
    - Default in Docker: `postgresql://postgres:postgres@uvm-postgres-cloud:5432/postgres`
  - `UVM_RAG_PG_TABLE_NAME` – pgvector table name.
    - Default: `uvm_vert_docs`
  - `UVM_RAG_EMBED_DIM` – embedding dimension (must match DB schema).
    - Default: `1024`

- Models:
  - `UVM_RAG_EMBED_MODEL`
    - Default: `Qwen/Qwen3-Embedding-0.6B`
  - `UVM_RAG_HF_MODEL`
    - Default: `Qwen/Qwen2.5-1.5B-Instruct`
  - `UVM_RAG_HF_TASK`
    - Default: `"text-generation"`

- Retrieval / answer sizes:
  - `UVM_RAG_RETRIEVER_TOP_K`
    - Default: `20`
  - `UVM_RAG_ANSWER_TOP_K`
    - Default: `8`

- Auto-warming:
  - `UVM_RAG_AUTOWARM`
    - If `"1"`, `backend.load_rag_pipeline()` is run at import time.
    - Docker sets this to `1` for you.

- Streamlit:
  - `STREAMLIT_SERVER_ADDRESS` – default `0.0.0.0` in Docker.
  - `STREAMLIT_SERVER_PORT` – default `8501` in Docker.

Ensure that:

- The embedding model and `UVM_RAG_EMBED_DIM` match the actual dimensionality of
  your stored vectors.
- `UVM_RAG_PG_TABLE_NAME` points to the correct table (e.g.,
  `uvm_rag_docs` if you rely on the dump in `uvm_db_dump.sql`).

---

## Prerequisites

To run the full system (Docker-based):

- NVIDIA GPU with a driver compatible with CUDA 11.8.
- Docker and NVIDIA Container Toolkit installed (so `--gpus all` works).
- A PostgreSQL 16+ server with the `vector` extension:
  - Can be a local container or external DB.
  - The provided `uvm_db_dump.sql` is designed for a server with `vector`
    installed.

To develop locally without Docker:

- Python 3.13 (or a version compatible with your environment).
- A CUDA-compatible PyTorch installation.
- PostgreSQL + pgvector extension accessible from your machine.

---

## Quickstart with Docker and a pre-built DB

This is the recommended way to get UVM-RAG running.

### 1. Restore the PostgreSQL database

1. Start (or create) a PostgreSQL instance with the `vector` extension. Example:

   ```bash
   docker run --name uvm-postgres-cloud \
     -e POSTGRES_PASSWORD=postgres \
     -p 5432:5432 \
     -d postgres:16
   ```

2. Copy `uvm_db_dump.sql` into the DB host and restore it. From your host:

   ```bash
   psql "postgresql://postgres:postgres@localhost:5432/postgres" \
     -f uvm_db_dump.sql
   ```

3. Make sure the DB is reachable from your app container. If you use a
   different host, port, or credentials, update `PG_CONN_STR` accordingly.

### 2. Build the GPU Docker image

From the project root:

```bash
docker build -t uvm-rag-streamlit:latest .
```

### 3. Run the app container

With Docker and a running PostgreSQL instance:

```bash
docker run --rm -it --gpus all \
  -e PG_CONN_STR="postgresql://postgres:postgres@uvm-postgres-cloud:5432/postgres" \
  -p 8501:8501 \
  uvm-rag-streamlit:latest
```

Then open the UI in a browser:

- http://localhost:8501

---

## Local development (without Docker)

You can also run the app directly on your machine.

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   # or:
   .venv\Scripts\activate           # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set the necessary environment variables (`PG_CONN_STR`, etc.), either via:

   - A `.env` file in the project root, or
   - Your shell environment.

4. Start the Streamlit UI:

   ```bash
   streamlit run app/ui_streamlit.py
   ```

5. Open http://localhost:8501 in your browser.

Remember that a CUDA-capable GPU and a reachable PostgreSQL/pgvector database
are still required.

---

## Ingestion pipeline (offline)

The ingestion notebooks in `ingest/` describe how to go from raw PDFs +
datasets to a populated pgvector store.

Typical flow:

1. **Parse and probe documents**
   - Use `01_toc_and_mineru_probe.ipynb` to:
     - Inspect structure of PDFs (TOC, headings, sections).
     - Evaluate MinerU’s parsing behavior.

2. **Post-process parsed content**
   - `01_1_postprocess_minerU.ipynb`:
     - Cleans up MinerU output.
     - Normalizes headings, sections, and text.

3. **Chunking**
   - `02_chunk_text.ipynb`:
     - Splits text into chunks suitable for dense retrieval (e.g., by section,
       paragraph, or token length).
     - Produces text + metadata that capture:
       - Standard, section title, type, URI/anchor, etc.

4. **pgvector schema**
   - `03_build_pgvector_index.ipynb`:
     - Creates the pgvector table and necessary indices.
     - Ensures the `vector` extension is installed and configured.

5. **Embedding and storing**
   - `03_1_embed_and_store.ipynb`:
     - Runs a document embedder (matching your chosen model).
     - Stores vectors + metadata into PostgreSQL via pgvector.

6. **Backend prototyping**
   - `04_rag_beckend.ipynb`:
     - Early prototype of the RAG pipeline, now encoded as Python in
       `app/backend.py`.

If you just want to query with the pre-built DB, you can skip these notebooks
and rely on the restored `uvm_db_dump.sql` instead.

---

## Using the UI

Once the app is running and accessible:

1. Open the Streamlit page (default http://localhost:8501).
2. Enter a question about UVM/SystemVerilog, for example:
   - *How does `uvm_packer` unpack dynamic arrays in UVM 1.2?*
3. Adjust sidebar sliders:
   - `Retriever top_k (PGVector)` – how many candidates to retrieve.
   - `Context docs in answer (N)` – how many documents to pass into the prompt.
4. Click **“Search and answer”**.
5. Read the answer and use:
   - The citation mapping table (`[i] → document/meta/URI`).
   - The expandable context documents to verify grounding.

If `show_debug` is enabled, you can inspect the raw pipeline output returned by
Haystack.

---

## Cloud deployment example (GCP VM)

The original workflow for running this project on a GCP VM looked like this.
You can adapt it to your own cloud or on-prem environment.

### 1. SSH into the VM (from your host)

```bash
gcloud compute ssh instance-20251208-052639
```

### 2. On the VM: download the DB dump

Install `gdown` and fetch the SQL dump (if not already present locally):

```bash
sudo apt install python3-pip

python3 -m venv gdown
source gdown/bin/activate

pip install gdown
gdown "https://drive.google.com/uc?id=1BoosDX35_0RIkG6WlLwAiVtJCyL4PFEE" -O uvm_db_dump.sql
```

### 3. On the VM: deploy PostgreSQL with pgvector

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

sudo docker start uvm-postgres-cloud    # or create it if it does not exist
```

Restore `uvm_db_dump.sql` into the running `uvm-postgres-cloud` PostgreSQL
instance (adjust connection details as needed).

### 4. On the VM: run the UVM-RAG app container

```bash
sudo docker run --rm -it --gpus all \
  -e PG_CONN_STR="postgresql://postgres:postgres@uvm-postgres-cloud:5432/postgres" \
  -p 8501:8501 \
  uvm-rag-streamlit:latest
```

Then point your browser (or an SSH tunnel) at port `8501` on the VM.

---

## Troubleshooting

- **No GPU available**
  - `backend.py` will raise `SystemExit("ERROR: CUDA GPU is required...")`.
  - Ensure:
    - You have an NVIDIA GPU.
    - Drivers and CUDA are installed.
    - Docker is configured with NVIDIA runtime (`--gpus all`).

- **Database connection errors**
  - Verify `PG_CONN_STR` and that:
    - PostgreSQL is running and reachable from the app container.
    - The `vector` extension is installed.
    - The expected table (`uvm_rag_docs` or `uvm_vert_docs`) and dimensions
      exist and match your configuration.

- **Model download / Hugging Face issues**
  - Ensure the container or host can reach Hugging Face to download Qwen
    models, or pre-mount a model cache via `HF_HOME`.


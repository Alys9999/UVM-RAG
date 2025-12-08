#!/usr/bin/env bash
set -e

echo "[entry] installing Python dependencies from requirements.txt..."
python3.13 -m pip install -r /app/requirements.txt

echo "[entry] cd /app/app and importing backend (auto-warm)..."
cd /app/app

# This will trigger load_rag_pipeline() once because UVM_RAG_AUTOWARM=1 in the env
python3.13 -c 'import backend'

echo "[entry] starting Streamlit UI on ${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}:${STREAMLIT_SERVER_PORT:-8501}..."
exec streamlit run ui_streamlit.py \
    --server.address="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
    --server.port="${STREAMLIT_SERVER_PORT:-8501}"

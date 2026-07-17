#!/bin/bash
# Container startup.
#
# 1. If the FAISS index isn't already on the mounted volume (fresh
#    machine or purged volume), build it once before serving. This
#    downloads ~500 arXiv abstracts and embeds them (~90s on the
#    reference machine).
#
# 2. Serve the FastAPI JSON API (uvicorn, :8001) alongside the Chainlit
#    UI (:8000). Both bind 0.0.0.0 — 127.0.0.1 wouldn't be reachable
#    through Docker port-forwarding.
set -e

if [ ! -f /app/index/faiss.index ]; then
    echo "[entrypoint] FAISS index not found, building one-time corpus (~90s)..."
    python -m rag.indexer
fi

echo "[entrypoint] Starting FastAPI JSON API (uvicorn) on 0.0.0.0:8001"
uvicorn backend.main:app --host 0.0.0.0 --port 8001 &

echo "[entrypoint] Launching Chainlit UI on 0.0.0.0:8000"
exec chainlit run frontend/app.py --host 0.0.0.0 --port 8000

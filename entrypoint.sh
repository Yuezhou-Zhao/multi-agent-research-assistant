#!/bin/bash
# entrypoint.sh — container startup.
#
# 1. If the FAISS index isn't already on the mounted volume (fresh
#    machine or purged volume), build it once before serving. This
#    downloads ~500 arXiv abstracts and embeds them (~90s on the M5
#    Pro reference machine). Section 7.2 Week 2's `python -m
#    rag.indexer` is the same command.
#
# 2. Launch Chainlit on 0.0.0.0:8000 (Section 7.2 Week 6). Never bind
#    to 127.0.0.1 in a container — Docker port-forwarding wouldn't
#    reach it.
set -e

if [ ! -f /app/index/faiss.index ]; then
    echo "[entrypoint] FAISS index not found, building one-time corpus (~90s)..."
    python -m rag.indexer
fi

echo "[entrypoint] Launching Chainlit on 0.0.0.0:8000"
exec chainlit run frontend/app.py --host 0.0.0.0 --port 8000

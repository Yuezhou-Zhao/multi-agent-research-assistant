# python:3.11-slim to match the venv the project developed against.
# -slim keeps image size down without giving up glibc.
FROM python:3.11-slim

# System deps: FlagEmbedding pulls in FAISS wheels that use libgomp1
# at runtime; without it the reranker crashes with "cannot open shared
# object file" the first time compute_score() is called.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements copied separately so `docker build` reuses the pip
# install layer whenever only source code changes. requirements.lock
# (pip freeze of the reference venv) pins every transitive version via
# pip's constraints mechanism for reproducible builds.
COPY requirements.txt requirements.lock ./
# Two-step install to survive PyPI throttling on the ~400MB torch wheel:
#   1. torch first, from PyTorch's own CPU-only index — the CPU wheel is
#      ~200MB (vs the ~426MB CUDA-enabled one on PyPI) AND download.
#      pytorch.org isn't behind the same throttle files.pythonhosted.org
#      applies to bulk downloads. Empirically the earlier build slowed
#      to 15 kB/s at the 213MB mark and timed out; this bypass fixes it.
#   2. remaining requirements next, with default-timeout + retries bumped
#      as a belt-and-suspenders for the smaller wheels.
#
# BuildKit cache mount: pip's HTTP cache is bind-mounted from the Docker
# builder's persistent cache, so re-runs of this RUN step (after a build
# interrupt, a network switch, or a requirements.txt tweak) reuse the
# wheels pip already downloaded instead of re-fetching hundreds of MB.
# --no-cache-dir is intentionally REMOVED so pip actually populates the
# cache; the mount is throwaway wrt the image so this doesn't bloat the
# built image.
# PyPI mirror override — the default files.pythonhosted.org CDN was
# throttling this route to ~1 kB/s and terminating connections
# mid-download, causing IncompleteRead on 6 MB wheels. Tsinghua's
# mirror is a full pypi.org sync (updates every few minutes) and
# typically serves 10-100x faster on Asia-Pacific routes. pypi.org is
# kept as --extra-index-url so any package Tsinghua happens not to
# have yet still resolves.
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_EXTRA_INDEX_URL=https://pypi.org/simple

# Three separate RUN layers so a flaky network resumes from the last
# completed layer (torch alone is ~200 MB) instead of restarting the
# whole install; the BuildKit cache mount de-duplicates downloads
# across rebuilds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --upgrade \
    --default-timeout=1000 \
    --retries=10 \
    --index-url ${PIP_INDEX_URL} \
    --extra-index-url ${PIP_EXTRA_INDEX_URL} \
    pip

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --default-timeout=1000 \
    --retries=10 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --default-timeout=1000 \
    --retries=10 \
    --index-url ${PIP_INDEX_URL} \
    --extra-index-url ${PIP_EXTRA_INDEX_URL} \
    -r requirements.txt -c requirements.lock

# Application source. tests/, scratchpad, and the built index are
# excluded via .dockerignore — index/ is mounted as a volume in
# docker-compose since it's regenerable, not baked in.
COPY backend/ ./backend/
COPY rag/ ./rag/
# Required, not optional: rag/mcp_client.py spawns `python -m
# mcp_servers.web_research` as a subprocess. Omit this and the web agent
# silently falls back to its in-process path on every query — degraded but
# green, which is the worst way to lose a component.
COPY mcp_servers/ ./mcp_servers/
COPY evaluation/ ./evaluation/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Persist the Hugging Face model cache to a mount point so the
# ~90MB bge-small-en-v1.5 + reranker downloads survive container
# recreation.
ENV HF_HOME=/hf_cache

# Chainlit UI (8000) + FastAPI JSON API (8001) — both started by
# entrypoint.sh.
EXPOSE 8000 8001

# API keys (OPENAI_API_KEY, TAVILY_API_KEY) come from docker-compose's
# env_file: .env directive at runtime. NEVER bake them into the image.

ENTRYPOINT ["./entrypoint.sh"]

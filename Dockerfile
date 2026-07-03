# python:3.11-slim to match the venv the project developed against
# (Section 7.3). -slim keeps image size down without giving up glibc.
FROM python:3.11-slim

# System deps: FlagEmbedding pulls in FAISS wheels that use libgomp1
# at runtime; without it the reranker crashes with "cannot open shared
# object file" the first time compute_score() is called.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt copied separately so `docker build` reuses the pip
# install layer whenever only source code changes.
COPY requirements.txt .
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
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade --default-timeout=1000 --retries 10 pip \
    && pip install --default-timeout=1000 --retries 10 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.1 \
    && pip install --default-timeout=1000 --retries 10 \
        -r requirements.txt

# Application source. tests/, scratchpad, and the built index are
# excluded via .dockerignore — index/ is mounted as a volume in
# docker-compose since it's regenerable (Week 2), not baked in.
COPY backend/ ./backend/
COPY rag/ ./rag/
COPY evaluation/ ./evaluation/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY entrypoint.sh ./
RUN chmod +x ./entrypoint.sh

# Persist the Hugging Face model cache to a mount point so the
# ~90MB bge-small-en-v1.5 + reranker downloads survive container
# recreation.
ENV HF_HOME=/hf_cache

# Chainlit UI port (Section 7.2 Week 6).
EXPOSE 8000

# API keys (OPENAI_API_KEY, TAVILY_API_KEY) come from docker-compose's
# env_file: .env directive at runtime. NEVER bake them into the image.

ENTRYPOINT ["./entrypoint.sh"]

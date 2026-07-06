"""Shared helpers: qwen client, robust JSON parsing, retries, JSONL I/O.

Kept dependency-light — the qwen endpoint is OpenAI-compatible (vLLM), so
we drive it with the stock `openai` AsyncOpenAI client pointed at a custom
base_url. gpt-4o-mini labeling reuses the project's SupervisorAgent so the
training labels are byte-identical to production behavior.
"""
import asyncio
import json
import re
from pathlib import Path

from openai import AsyncOpenAI

from . import config

# ── Clients ──────────────────────────────────────────────────────────────
_qwen_client: AsyncOpenAI | None = None


def qwen_client() -> AsyncOpenAI:
    """Lazy singleton AsyncOpenAI pointed at the qwen vLLM endpoint."""
    global _qwen_client
    if _qwen_client is None:
        config.require_qwen()
        _qwen_client = AsyncOpenAI(
            base_url=config.QWEN_BASE_URL,
            api_key=config.QWEN_API_KEY,
        )
    return _qwen_client


# qwen3.5-397b-a17b is a reasoning model: left in thinking mode it spends
# the whole token budget on an internal trace and returns content=None with
# finish_reason=length (observed on web_only/ambiguous generation prompts).
# Neither query generation nor routing classification needs chain-of-thought,
# so we disable it. This vLLM deployment honors chat_template_kwargs but NOT
# the /no_think soft-switch, so we pass it via extra_body.
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


async def qwen_chat(prompt: str, *, temperature: float, max_tokens: int = 2048) -> str:
    """One qwen chat completion (thinking disabled) with retry/backoff.
    Returns raw text."""
    client = qwen_client()

    async def _call() -> str:
        resp = await client.chat.completions.create(
            model=config.QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=_NO_THINK,
        )
        return resp.choices[0].message.content or ""

    return await with_retries(_call)


async def with_retries(coro_factory):
    """Run an async callable with exponential backoff on any exception."""
    last = None
    for attempt in range(config.MAX_RETRIES):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001 — transient network/API errors
            last = e
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(config.RETRY_BASE_DELAY * (2 ** attempt))
    raise last


# ── Robust JSON extraction (vLLM models often wrap JSON in prose/fences) ──
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str):
    """Best-effort parse of a JSON object or array out of a model reply.

    Handles: raw JSON, ```json fenced blocks, and JSON embedded in prose
    (grabs the first balanced {...} or [...] span). Raises ValueError if
    nothing parses."""
    text = text.strip()
    # 1. straight parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. fenced block
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 3. first balanced array or object span
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No parseable JSON in reply: {text[:200]!r}")


# ── Routing-label helpers ────────────────────────────────────────────────
def route_of(use_arxiv: bool, use_web: bool) -> str:
    """Collapse the (use_arxiv, use_web) pair into a single route label."""
    if use_arxiv and use_web:
        return "both"
    if use_arxiv:
        return "arxiv"
    if use_web:
        return "web"
    return "neither"


def normalize_query(q: str) -> str:
    """Lowercase + collapse whitespace + strip trailing punctuation, for
    exact-duplicate detection."""
    return re.sub(r"\s+", " ", q.lower()).strip().rstrip("?.!")


# ── JSONL I/O with resume support ────────────────────────────────────────
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(path: Path) -> set[str]:
    """IDs already present in an output JSONL — lets a labeling run resume
    after an interruption without re-calling the API on finished rows."""
    return {r["id"] for r in read_jsonl(path) if "id" in r}

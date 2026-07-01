"""Rule-based citation grounding — Critic layer 2 (Section 4.7 diagram).

Zero LLM calls, purely structural checks:
  1. Every `[N]` marker in the draft must be an integer 1..len(chunks).
     This is the ALWAYS-FAIL rule for out-of-range markers — but with the
     index-based citation architecture (Section 4.8), out-of-range is
     the only failure mode possible; the LLM cannot fabricate an
     "identifier that doesn't exist" because the whitelist is just the
     integers 1..N. The dangling-citation report exists to catch:
       - Writer picking an index > N (rare with a small N in the prompt)
       - Non-integer content in brackets ("[N]" containing letters, or
         formats the prompt banned like "[1, 3]")
  2. At most 1 sentence in the draft may lack a `[N]` marker. Set to 1
     rather than 0 empirically: Writer routinely produces an uncited
     intro sentence ("Chain of thought reasoning significantly
     enhances...") followed by well-cited body sentences. Requiring
     EVERY sentence to have a citation drove 100% force_finalized in
     early testing.

MAX_UNCITED_SENTENCES = 1 is a per-draft tolerance, not a percentage.

Design goal (Section 4.7): a cheap prefilter that catches obvious
grounding failures before the L3 LLM judge fires. False positives route
to L3 anyway; false negatives (Critic OKs an ungrounded draft) are the
failure mode we minimize.
"""
import re
from dataclasses import dataclass, field

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

MAX_UNCITED_SENTENCES = 1  # empirical tolerance — see module docstring


@dataclass
class CitationReport:
    passed: bool
    # Raw bracket contents in the draft that could not be resolved to a
    # 1-based index in [1, n_chunks]. Reported for the rollback feedback.
    dangling_citations: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    # 1-based indices that WERE valid (i.e., appeared in the draft and
    # were resolvable). Downstream (Critic L3, finalize) uses this to
    # decide which chunks to hand the LLM judge, and to decide which
    # source ids end up in the final citations list.
    cited_indices: set[int] = field(default_factory=set)
    sanitized_draft: str = ""  # dangling brackets removed

    def summary(self) -> str:
        parts = []
        if self.dangling_citations:
            parts.append(f"invalid citation markers: {sorted(self.dangling_citations)}")
        if self.uncited_sentences:
            n = len(self.uncited_sentences)
            parts.append(f"{n} sentence(s) missing citations after sanitization")
        return "; ".join(parts) if parts else "citations OK"


def _strip_marker(draft: str, marker_content: str) -> str:
    """Remove all occurrences of [{marker_content}] from the draft, plus
    the leading space, so "text [X]." collapses cleanly to "text.".
    """
    escaped = re.escape(marker_content)
    return re.sub(rf"\s*\[{escaped}\]", "", draft)


def check_citations(draft: str, merged_chunks: list[dict]) -> CitationReport:
    """Check draft [N] citations against the numbered chunk list.

    Behavior (Section 4.8):
      1. Any `[N]` where N is a positive integer in [1, len(chunks)] is
         valid. Anything else in a bracket (non-integer, out-of-range,
         "1, 2" format) is flagged as dangling and stripped from the
         sanitized draft.
      2. On the sanitized draft, apply the MAX_UNCITED_SENTENCES
         tolerance. If more than that many sentences are left uncited,
         roll back to Writer.

    merged_chunks entries must have a "source" key (used later by
    finalizer_node to resolve the indices to real ids).
    """
    n_chunks = len(merged_chunks)

    all_markers = _CITATION_RE.findall(draft)
    valid_indices: set[int] = set()
    dangling: list[str] = []
    for raw in all_markers:
        stripped = raw.strip()
        try:
            idx = int(stripped)
        except ValueError:
            if stripped not in dangling:
                dangling.append(stripped)
            continue
        if 1 <= idx <= n_chunks:
            valid_indices.add(idx)
        else:
            if stripped not in dangling:
                dangling.append(stripped)

    sanitized = draft
    for content in dangling:
        sanitized = _strip_marker(sanitized, content)

    uncited_sentences: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(sanitized.strip()):
        stripped_s = sentence.strip()
        if not stripped_s:
            continue
        if not _CITATION_RE.search(stripped_s):
            uncited_sentences.append(stripped_s)

    passed = len(uncited_sentences) <= MAX_UNCITED_SENTENCES
    return CitationReport(
        passed=passed,
        dangling_citations=dangling,
        uncited_sentences=uncited_sentences,
        cited_indices=valid_indices,
        sanitized_draft=sanitized,
    )


def resolve_citations(draft: str, merged_chunks: list[dict]) -> tuple[str, list[str]]:
    """Rewrite `[N]` markers into `[source_id]` markers, using the same
    1..N mapping the Writer saw. Returns (resolved_draft, ordered_ids).

    Called from finalizer_node (Section 4.6): the user-visible
    final_answer displays real arXiv ids, not the internal indices.
    """
    n_chunks = len(merged_chunks)
    ordered_ids: list[str] = []
    seen: set[str] = set()

    def sub(match: re.Match) -> str:
        raw = match.group(1).strip()
        try:
            idx = int(raw)
        except ValueError:
            return match.group(0)  # leave garbage brackets alone
        if not (1 <= idx <= n_chunks):
            return match.group(0)
        source_id = merged_chunks[idx - 1]["source"]
        if source_id not in seen:
            seen.add(source_id)
            ordered_ids.append(source_id)
        return f"[{source_id}]"

    resolved = _CITATION_RE.sub(sub, draft)
    return resolved, ordered_ids

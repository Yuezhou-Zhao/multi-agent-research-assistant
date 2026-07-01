"""Rule-based citation grounding — Critic layer 2 (Section 4.7 diagram).

Zero LLM calls, purely structural checks:
  1. Every `[X]` marker in the draft must reference a real source id in
     the merged_chunks pool.
  2. Every sentence that makes a citation-worthy claim must carry at
     least one `[X]` marker. "Citation-worthy" here is a coarse
     heuristic — any sentence in the answer body — since fine-grained
     claim detection would need an LLM, defeating the point of a
     zero-LLM check.
  3. Every source id present in merged_chunks that gets cited must
     survive the check (i.e., we don't require every source to be cited,
     but every cited source must exist).

This is deliberately conservative: the design goal (Section 4.7) is a
cheap prefilter that catches obvious grounding failures before the L3
LLM judge fires. False positives (Critic flags a fine draft) route to
the LLM judge anyway; false negatives (Critic OKs an ungrounded draft)
are the failure mode we want to minimize, so the rules err on the strict
side.
"""
import re
from dataclasses import dataclass, field

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitationReport:
    passed: bool
    dangling_citations: list[str] = field(default_factory=list)  # ids in draft not in sources
    uncited_sentences: list[str] = field(default_factory=list)  # body sentences with no [X]
    cited_source_ids: set[str] = field(default_factory=set)

    def summary(self) -> str:
        parts = []
        if self.dangling_citations:
            parts.append(f"dangling citations: {sorted(self.dangling_citations)}")
        if self.uncited_sentences:
            n = len(self.uncited_sentences)
            parts.append(f"{n} sentence(s) missing citations")
        return "; ".join(parts) if parts else "citations OK"


def check_citations(draft: str, merged_chunks: list[dict]) -> CitationReport:
    """Check draft citations against the source pool.

    merged_chunks entries must have a "source" key (the arxiv id or url
    used as the citation marker).
    """
    valid_source_ids = {c["source"] for c in merged_chunks}

    all_markers = _CITATION_RE.findall(draft)
    cited_ids = {m.strip() for m in all_markers}
    dangling = sorted(cited_ids - valid_source_ids)

    uncited_sentences: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(draft.strip()):
        stripped = sentence.strip()
        if not stripped:
            continue
        if not _CITATION_RE.search(stripped):
            uncited_sentences.append(stripped)

    passed = not dangling and not uncited_sentences
    return CitationReport(
        passed=passed,
        dangling_citations=dangling,
        uncited_sentences=uncited_sentences,
        cited_source_ids=cited_ids & valid_source_ids,
    )

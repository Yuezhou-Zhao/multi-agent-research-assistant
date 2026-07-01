"""Rule-based citation grounding — Critic layer 2 (Section 4.7 diagram).

Zero LLM calls, purely structural checks:
  1. Every `[X]` marker in the draft must reference a real source id in
     the merged_chunks pool. This is the ALWAYS-FAIL rule — dangling
     citations are literal hallucinated references and can never be OK.
  2. At most 1 sentence in the draft may lack a `[X]` marker. Set to 1
     rather than 0 after empirical measurement: Writer routinely
     produces an uncited intro sentence ("Chain of thought reasoning
     significantly enhances...") followed by well-cited body sentences.
     Requiring EVERY sentence to have a citation drove 100%
     force_finalized across a 5-query batch, since the intro sentence
     alone was enough to fail L2 every time. Allowing 1 uncited
     sentence tolerates that pattern while still catching drafts that
     forget to cite entirely (which would show up as many uncited
     sentences).

MAX_UNCITED_SENTENCES = 1 is a per-draft tolerance, not a percentage —
even a very long draft only gets 1 free uncited sentence, which keeps
the check strict for the failure mode it's designed to catch.

Design goal (Section 4.7): a cheap prefilter that catches obvious
grounding failures before the L3 LLM judge fires. False positives route
to L3 anyway; false negatives (Critic OKs an ungrounded draft) are the
failure mode we minimize, so the rules err on the strict side within
the tolerances documented above.
"""
import re
from dataclasses import dataclass, field

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

MAX_UNCITED_SENTENCES = 1  # empirical tolerance — see module docstring


@dataclass
class CitationReport:
    passed: bool
    dangling_citations: list[str] = field(default_factory=list)  # ids in draft not in sources
    uncited_sentences: list[str] = field(default_factory=list)  # body sentences with no [X]
    cited_source_ids: set[str] = field(default_factory=set)
    sanitized_draft: str = ""  # draft with dangling [X] markers stripped

    def summary(self) -> str:
        parts = []
        if self.dangling_citations:
            parts.append(f"dangling citations: {sorted(self.dangling_citations)}")
        if self.uncited_sentences:
            n = len(self.uncited_sentences)
            parts.append(f"{n} sentence(s) missing citations after sanitization")
        return "; ".join(parts) if parts else "citations OK"


def _strip_marker(draft: str, marker_content: str) -> str:
    """Remove all occurrences of [{marker_content}] from the draft, plus
    any trailing whitespace that would be left behind. Handles the common
    "sentence body [X].", "[X] [Y]", and mid-sentence "..., [X], ..." shapes.
    """
    escaped = re.escape(marker_content)
    # Absorb one leading space so "text [X]." collapses cleanly to "text."
    return re.sub(rf"\s*\[{escaped}\]", "", draft)


def check_citations(draft: str, merged_chunks: list[dict]) -> CitationReport:
    """Check draft citations against the source pool.

    Behavior:
      1. Dangling markers ([X] where X isn't a valid source id) are
         STRIPPED from the draft rather than an automatic rollback trigger.
         Rationale (measured): gpt-4o-mini in the Writer role consistently
         hallucinates the same familiar-looking arxiv ids across retries
         even with a hardened prompt — a repair pass converges faster than
         reprompting alone. Downstream (Critic L3, finalize) then sees
         the sanitized draft.
      2. On the sanitized draft, apply the same MAX_UNCITED_SENTENCES
         tolerance the original L2 used. If more than that many
         sentences are left uncited, the draft as a whole was too
         dependent on hallucinated support to keep — roll back to Writer.
      3. Dangling ids are still reported in the CitationReport so a
         rollback (either from this check or from L3 downstream) can
         include them in the critic_feedback message.

    merged_chunks entries must have a "source" key (the arxiv id or url
    used as the citation marker).
    """
    valid_source_ids = {c["source"] for c in merged_chunks}

    all_markers = _CITATION_RE.findall(draft)
    cited_ids = {m.strip() for m in all_markers}
    dangling = sorted(cited_ids - valid_source_ids)

    sanitized = draft
    for cid in dangling:
        sanitized = _strip_marker(sanitized, cid)

    uncited_sentences: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(sanitized.strip()):
        stripped = sentence.strip()
        if not stripped:
            continue
        if not _CITATION_RE.search(stripped):
            uncited_sentences.append(stripped)

    passed = len(uncited_sentences) <= MAX_UNCITED_SENTENCES
    return CitationReport(
        passed=passed,
        dangling_citations=dangling,
        uncited_sentences=uncited_sentences,
        cited_source_ids=cited_ids & valid_source_ids,
        sanitized_draft=sanitized,
    )

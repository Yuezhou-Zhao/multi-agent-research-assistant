"""ParentChildChunker — hand-written, no framework dependency.

Resolves the retrieval-generation tension:
  Child (128 tokens, word-approx) -> FAISS index -> high embedding coherence
    -> precise retrieval
  Parent (512 tokens, word-approx) -> dict lookup -> full context
    -> coherent generation

This is deliberately ~50 lines of plain Python instead of a framework
abstraction: the point of this project is retrieval-generation tension I
can fully explain, not a black-box chunking library.

Hash-prefixed IDs prevent collision across documents:
  parent_id = f"{md5(doc_id)[:8]}_p{i}"
  child_id  = f"{parent_id}_c{j}"
"""
import hashlib


class ParentChildChunker:
    def __init__(self, parent_tokens: int = 512, child_tokens: int = 128, overlap: int = 20):
        if overlap >= child_tokens:
            raise ValueError("overlap must be smaller than child_tokens or _split never advances")
        self.parent_tokens = parent_tokens
        self.child_tokens = child_tokens
        self.overlap = overlap
        self.parent_store: dict[str, str] = {}
        self.child_to_parent: dict[str, str] = {}

    def chunk_document(self, doc_id: str, text: str) -> list[tuple[str, str]]:
        """Split `text` into parent chunks, then each parent into overlapping
        child chunks. Returns the (child_id, child_text) pairs for indexing;
        parents are stored internally and fetched via get_parent().
        """
        doc_hash = hashlib.md5(doc_id.encode()).hexdigest()
        parents = self._split(text, self.parent_tokens, overlap=50)
        children = []
        for i, parent_text in enumerate(parents):
            pid = f"{doc_hash[:8]}_p{i}"
            self.parent_store[pid] = parent_text
            for j, child_text in enumerate(
                self._split(parent_text, self.child_tokens, self.overlap)
            ):
                cid = f"{pid}_c{j}"
                self.child_to_parent[cid] = pid
                children.append((cid, child_text))
        return children

    def get_parent(self, child_id: str) -> str:
        return self.parent_store[self.child_to_parent[child_id]]

    @staticmethod
    def _split(text: str, size: int, overlap: int) -> list[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + size]))
            i += size - overlap
        return chunks or [text]

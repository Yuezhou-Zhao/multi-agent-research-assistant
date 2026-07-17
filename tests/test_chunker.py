"""Tests for the hand-written ParentChildChunker.

Focus: hash-prefixed ID collision avoidance and parent/child round-trip
correctness — the two properties easiest to get subtly wrong.
"""
import pytest

from rag.chunker import ParentChildChunker


def make_words(n: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


class TestSplit:
    def test_split_respects_size(self):
        text = make_words(300)
        chunks = ParentChildChunker._split(text, size=128, overlap=20)
        assert all(len(c.split()) <= 128 for c in chunks)

    def test_split_overlap_shares_boundary_words(self):
        text = make_words(300)
        chunks = ParentChildChunker._split(text, size=128, overlap=20)
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        # last `overlap` words of chunk 0 should equal first `overlap` words of chunk 1
        assert first_words[-20:] == second_words[:20]

    def test_split_short_text_returns_single_chunk(self):
        text = make_words(10)
        chunks = ParentChildChunker._split(text, size=128, overlap=20)
        assert chunks == [text]

    def test_split_empty_text_returns_nonempty_list(self):
        chunks = ParentChildChunker._split("", size=128, overlap=20)
        assert chunks == [""]


class TestConstructorValidation:
    def test_overlap_equal_to_child_tokens_raises(self):
        with pytest.raises(ValueError):
            ParentChildChunker(child_tokens=128, overlap=128)

    def test_overlap_greater_than_child_tokens_raises(self):
        with pytest.raises(ValueError):
            ParentChildChunker(child_tokens=128, overlap=200)

    def test_valid_overlap_constructs(self):
        chunker = ParentChildChunker(child_tokens=128, overlap=20)
        assert chunker.child_tokens == 128


class TestHashIDCollision:
    def test_different_docs_produce_different_parent_prefixes(self):
        chunker = ParentChildChunker(parent_tokens=512, child_tokens=128, overlap=20)
        chunker.chunk_document("doc-a", make_words(600))
        chunker.chunk_document("doc-b", make_words(600))

        prefixes = {pid.split("_p")[0] for pid in chunker.parent_store}
        assert len(prefixes) == 2

    def test_same_doc_id_is_deterministic(self):
        chunker1 = ParentChildChunker()
        chunker2 = ParentChildChunker()
        children1 = chunker1.chunk_document("doc-x", make_words(600))
        children2 = chunker2.chunk_document("doc-x", make_words(600))
        assert [cid for cid, _ in children1] == [cid for cid, _ in children2]

    def test_child_ids_globally_unique_across_documents(self):
        chunker = ParentChildChunker(parent_tokens=512, child_tokens=128, overlap=20)
        children_a = chunker.chunk_document("doc-a", make_words(600))
        children_b = chunker.chunk_document("doc-b", make_words(600))
        ids_a = {cid for cid, _ in children_a}
        ids_b = {cid for cid, _ in children_b}
        assert ids_a.isdisjoint(ids_b)

    def test_child_id_is_prefixed_by_its_parent_id(self):
        chunker = ParentChildChunker(parent_tokens=512, child_tokens=128, overlap=20)
        children = chunker.chunk_document("doc-a", make_words(600))
        for cid, _ in children:
            parent_id = chunker.child_to_parent[cid]
            assert cid.startswith(parent_id + "_c")


class TestRoundTripCorrectness:
    def test_get_parent_returns_text_containing_child(self):
        chunker = ParentChildChunker(parent_tokens=512, child_tokens=128, overlap=20)
        children = chunker.chunk_document("doc-a", make_words(600))
        for cid, child_text in children:
            parent_text = chunker.get_parent(cid)
            assert child_text.split()[0] in parent_text.split()

    def test_all_children_map_to_a_stored_parent(self):
        chunker = ParentChildChunker(parent_tokens=512, child_tokens=128, overlap=20)
        children = chunker.chunk_document("doc-a", make_words(600))
        for cid, _ in children:
            pid = chunker.child_to_parent[cid]
            assert pid in chunker.parent_store

    def test_children_cover_full_parent_without_gaps(self):
        """Every word in a parent chunk should appear in at least one child
        (allowing for overlap) — no silent data loss during child-splitting."""
        chunker = ParentChildChunker(parent_tokens=512, child_tokens=128, overlap=20)
        children = chunker.chunk_document("doc-a", make_words(300))

        by_parent: dict[str, list[str]] = {}
        for cid, child_text in children:
            by_parent.setdefault(chunker.child_to_parent[cid], []).append(child_text)

        for pid, parent_text in chunker.parent_store.items():
            covered_words: set[str] = set()
            for child_text in by_parent[pid]:
                covered_words.update(child_text.split())
            assert set(parent_text.split()) <= covered_words

    def test_multiple_documents_isolated_in_parent_store(self):
        chunker = ParentChildChunker(parent_tokens=512, child_tokens=128, overlap=20)
        chunker.chunk_document("doc-a", "alpha " * 600)
        chunker.chunk_document("doc-b", "beta " * 600)
        for pid, text in chunker.parent_store.items():
            words = set(text.split())
            assert words <= {"alpha"} or words <= {"beta"}

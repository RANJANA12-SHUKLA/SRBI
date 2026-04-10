import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import retrieval
from retrieval import build_index, retrieve
from schema import ChunkWithMeta


def test_retrieve_returns_relevant_chunks_and_honors_source_filter():
    def fake_embed_texts(texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "visitor" in lowered or "plants" in lowered or "manual" in lowered:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 1.0, 0.0, 0.0])
        return vectors

    retrieval.embed_texts = fake_embed_texts

    chunks = []
    for idx in range(10):
        chunks.append(
            ChunkWithMeta(
                text=f"Plant operations team uses manual visitor management workflows at facility {idx}.",
                source_url=f"https://primary.example.com/source-{idx}",
                source_type="PRIMARY",
                scraped_at="2026-03-01",
                chunk_index=idx,
            )
        )
    for idx in range(10, 20):
        chunks.append(
            ChunkWithMeta(
                text=f"Consumer marketing campaign update and brand launch story {idx}.",
                source_url=f"https://secondary.example.com/source-{idx}",
                source_type="SECONDARY",
                scraped_at="2026-03-01",
                chunk_index=idx,
            )
        )

    index, metadata = build_index(chunks)

    results = retrieve("visitor management across plants", index, metadata, top_k=3)
    assert results
    assert any("visitor management" in item.text.lower() for item in results)

    filtered = retrieve(
        "visitor management across plants",
        index,
        metadata,
        top_k=5,
        source_type_filter="PRIMARY",
    )
    assert filtered
    assert all(item.source_type == "PRIMARY" for item in filtered)

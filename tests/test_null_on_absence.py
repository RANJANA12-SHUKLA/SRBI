import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extraction
from extraction import extract_funding
from schema import ChunkWithMeta


def test_extract_funding_returns_nulls_when_context_has_no_funding(monkeypatch):
    monkeypatch.setattr(extraction, "client", None)
    context = [
        ChunkWithMeta(
            text="The company opened a new manufacturing unit and hired a VP of operations.",
            source_url="https://example.com/press-release",
            source_type="SECONDARY",
            scraped_at="2026-03-20",
            chunk_index=0,
        )
    ]

    result = extract_funding(context)

    assert result.total_raised is None
    assert result.last_round_amount is None
    assert result.last_round_date is None
    assert result.pe_backed is None
    assert result.lead_investors == []
    assert result.source_urls == []

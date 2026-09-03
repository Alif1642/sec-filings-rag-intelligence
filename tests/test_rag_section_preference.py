from types import SimpleNamespace

from src.rag.pipeline import RAGPipeline


def test_risk_query_prefers_item_1a_when_enough_hits_exist():
    hits = [
        SimpleNamespace(chunk={"section": "Item 1A. Risk Factors"}),
        SimpleNamespace(chunk={"section": "Item 1A. Risk Factors"}),
        SimpleNamespace(chunk={"section": "Item 16"}),
        SimpleNamespace(chunk={"section": "Item 1A. Risk Factors"}),
    ]

    filtered = RAGPipeline._apply_section_preference(
        "What are the company's major risk factors?",
        hits,
    )

    assert len(filtered) == 3
    assert all(
        hit.chunk["section"].startswith("Item 1A")
        for hit in filtered
    )

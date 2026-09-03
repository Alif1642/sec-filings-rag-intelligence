from ingestion.chunker import StructureAwareChunker
from ingestion.filing_parser import DocumentBlock
from src.config import Settings


def test_structure_aware_chunking_preserves_metadata(tmp_path):
    settings = Settings(data_dir=tmp_path, chunk_target_tokens=120, chunk_overlap_tokens=20, chunk_min_tokens=20)
    blocks = [
        DocumentBlock(0, 'heading', 'Item 1. Business', 'Item 1. Business'),
        DocumentBlock(1, 'paragraph', 'alpha ' * 60, 'Item 1. Business'),
        DocumentBlock(2, 'heading', 'Item 1A. Risk Factors', 'Item 1A. Risk Factors'),
        DocumentBlock(3, 'paragraph', 'risk ' * 60, 'Item 1A. Risk Factors'),
    ]
    meta = {
        'ticker': 'AAPL', 'cik': '0000320193', 'form': '10-K',
        'filingDate': '2025-10-31', 'reportDate': '2025-09-27',
        'accessionNumber': '0000320193-25-000001', 'filing_url': 'https://www.sec.gov/example'
    }
    chunks = StructureAwareChunker(settings).chunk(blocks, meta)
    assert len(chunks) >= 2
    assert {c.section for c in chunks} >= {'Item 1. Business', 'Item 1A. Risk Factors'}
    assert all(c.ticker == 'AAPL' and c.chunk_id for c in chunks)

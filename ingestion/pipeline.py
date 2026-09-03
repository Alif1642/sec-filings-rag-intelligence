from __future__ import annotations

import argparse
import json

from ingestion.chunker import StructureAwareChunker
from ingestion.filing_downloader import FilingDownloader
from ingestion.filing_parser import FilingParser
from src.config import get_settings


class IngestionPipeline:
    """End-to-end filing download, parse and structure-aware chunk pipeline."""

    def __init__(self):
        self.settings = get_settings()
        self.downloader = FilingDownloader()
        self.parser = FilingParser()
        self.chunker = StructureAwareChunker()

    def ingest(self, ticker: str, form: str = '10-K', filing_date: str | None = None) -> dict:
        meta, raw_path = self.downloader.download(ticker, form=form, filing_date=filing_date)
        raw_html = raw_path.read_text(encoding='utf-8')
        blocks = self.parser.parse(raw_html)
        chunks = self.chunker.chunk(blocks, meta)
        accession = meta['accessionNumber'].replace('-', '')
        out_dir = self.settings.data_dir / 'processed' / meta['ticker'] / accession
        out_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = out_dir / 'chunks.json'
        chunks_path.write_text(json.dumps([c.to_dict() for c in chunks], indent=2), encoding='utf-8')
        manifest = {
            'metadata': meta,
            'raw_path': str(raw_path),
            'chunks_path': str(chunks_path),
            'block_count': len(blocks),
            'chunk_count': len(chunks),
        }
        (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Ingest a SEC filing')
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--form', default='10-K', choices=['10-K', '10-Q'])
    parser.add_argument('--filing-date', default=None)
    args = parser.parse_args()
    manifest = IngestionPipeline().ingest(args.ticker, args.form, args.filing_date)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

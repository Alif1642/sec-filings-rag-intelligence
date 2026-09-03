from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import get_settings
from src.retrieval.vector_store import FAISSVectorStore

parser = argparse.ArgumentParser(description='Build a FAISS index from processed SEC chunks')
parser.add_argument('--chunks', default=None, help='Path to chunks.json; defaults to newest processed file')
args = parser.parse_args()
settings = get_settings()
if args.chunks:
    chunks_path = Path(args.chunks)
else:
    candidates = sorted((settings.data_dir / 'processed').glob('*/*/chunks.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit('No processed chunks found. Run ingestion first.')
    chunks_path = candidates[0]
chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
store = FAISSVectorStore(settings)
store.build(chunks)
name = chunks_path.parent.name
out = settings.data_dir / 'indexes' / name
store.save(out)
print(f'Built index with {len(chunks)} chunks at {out}')

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import Settings, get_settings
from src.retrieval.bm25 import RetrievalHit


class VectorIndex(Protocol):
    """Vector retrieval boundary; FAISS is local default, pgvector can implement this contract."""

    def build(self, chunks: list[dict]) -> None: ...

    def search(self, query: str, top_k: int = 20) -> list[RetrievalHit]: ...


class VectorStoreError(RuntimeError):
    pass


class FAISSVectorStore:
    """FAISS cosine-similarity index with lazy SentenceTransformer loading."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model = None
        self.index = None
        self.chunks: list[dict] = []
        self.dimension: int | None = None

    def _load_dependencies(self):
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise VectorStoreError('Install project ML dependencies to use dense retrieval.') from exc
        if self.model is None:
            self.model = SentenceTransformer(self.settings.embedding_model)
        return faiss

    def _embed(self, texts: list[str]) -> np.ndarray:
        self._load_dependencies()
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype='float32')

    def build(self, chunks: list[dict]) -> None:
        faiss = self._load_dependencies()
        self.chunks = list(chunks)
        if not self.chunks:
            self.index = None
            self.dimension = None
            return
        vectors = self._embed([c.get('text', '') for c in self.chunks])
        self.dimension = int(vectors.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 20) -> list[RetrievalHit]:
        if self.index is None or not self.chunks:
            return []
        query_vector = self._embed([query])
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vector, k)
        hits: list[RetrievalHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False), start=1):
            if idx < 0:
                continue
            hits.append(RetrievalHit(self.chunks[int(idx)], float(score), rank, 'dense'))
        return hits

    def save(self, directory: str | Path) -> None:
        if self.index is None:
            raise VectorStoreError('Index has not been built')
        faiss = self._load_dependencies()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / 'index.faiss'))
        (directory / 'chunks.json').write_text(json.dumps(self.chunks), encoding='utf-8')
        (directory / 'meta.json').write_text(json.dumps({'embedding_model': self.settings.embedding_model, 'dimension': self.dimension}), encoding='utf-8')

    def load(self, directory: str | Path) -> None:
        faiss = self._load_dependencies()
        directory = Path(directory)
        self.index = faiss.read_index(str(directory / 'index.faiss'))
        self.chunks = json.loads((directory / 'chunks.json').read_text(encoding='utf-8'))
        self.dimension = int(self.index.d)

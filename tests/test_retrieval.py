import numpy as np
import pytest

from src.config import Settings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.vector_store import FAISSVectorStore


def chunks():
    return [
        {'chunk_id': 'a', 'text': 'revenue increased due to services growth'},
        {'chunk_id': 'b', 'text': 'risk factors include supply chain concentration'},
        {'chunk_id': 'c', 'text': 'cash flow from operating activities was strong'},
    ]


def test_bm25_retrieval():
    hits = BM25Retriever(chunks()).search('supply chain risk', top_k=2)
    assert hits[0].chunk['chunk_id'] == 'b'


def test_rrf_fusion():
    retriever = BM25Retriever(chunks())
    first = retriever.search('revenue', top_k=3)
    second = retriever.search('growth revenue', top_k=3)
    fused = reciprocal_rank_fusion([first, second], top_k=3)
    assert fused[0].chunk['chunk_id'] == 'a'


def test_faiss_vector_store_without_model_download(tmp_path, monkeypatch):
    faiss = pytest.importorskip('faiss')
    settings = Settings(data_dir=tmp_path)
    store = FAISSVectorStore(settings)
    monkeypatch.setattr(store, '_load_dependencies', lambda: faiss)

    def fake_embed(texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if 'risk' in lowered or 'supply' in lowered:
                vectors.append([1.0, 0.0])
            elif 'revenue' in lowered or 'growth' in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.7, 0.7])
        arr = np.asarray(vectors, dtype='float32')
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return arr

    monkeypatch.setattr(store, '_embed', fake_embed)
    store.build(chunks())
    hits = store.search('risk exposure', top_k=1)
    assert hits[0].chunk['chunk_id'] == 'b'

# Retrieval Design

## Structure-aware chunking

The parser extracts ordered headings, paragraphs and tables. Chunk assembly respects filing sections and paragraph boundaries instead of blindly splitting every N characters. Defaults target 700 estimated tokens with 80-token overlap and are configurable by environment variables.

Each chunk carries ticker, CIK, form, filing/report dates, accession number, section, source URL, anchor and a stable chunk ID. That metadata is preserved through retrieval so citations can be generated from real filing provenance.

## Sparse retrieval

`rank-bm25` indexes tokenized filing chunks. Sparse retrieval is useful for exact financial terminology, section names, risk language and uncommon entities.

## Dense retrieval

`sentence-transformers` produces normalized embeddings using `BAAI/bge-small-en-v1.5` by default. FAISS `IndexFlatIP` performs cosine-equivalent similarity search over normalized vectors.

## Hybrid fusion

BM25 and dense rankings are combined using Reciprocal Rank Fusion:

`score = Σ 1 / (k + rank)`

The default candidate set is 20 chunks.

## Reranking

A cross encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks candidate query/passage pairs and keeps the top six by default. Reranking can be disabled for lower-latency experiments.

## Prompt-injection boundary

SEC filing content is treated as untrusted evidence. Instruction-like document text is flagged during parsing, and the generation system prompt explicitly prohibits following instructions contained inside filings.

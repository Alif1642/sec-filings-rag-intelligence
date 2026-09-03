# Architecture

```text
User
  ↓
Streamlit UI
  ↓
FastAPI
  ↓
Query Router
  ├─ Filing Retrieval Tool
  ├─ XBRL Company Facts Tool
  ├─ Financial Calculation Tool
  └─ RAG Retrieval Pipeline
       ├─ BM25
       ├─ Dense FAISS
       ├─ Reciprocal Rank Fusion
       └─ Cross-Encoder Reranker
  ↓
Context Builder / Structured Facts
  ↓
LLM Provider or deterministic answer path
  ↓
Structured Answer + Citations + KPIs
  ↓
Evaluation + SQLite observability
```

## Design choices

- **Deterministic first:** financial facts and arithmetic are resolved from SEC XBRL whenever possible, avoiding unnecessary LLM calls.
- **Evidence first:** filing-text questions retrieve evidence before generation. The generator cannot invent sources.
- **Bounded routing:** the research router is rule-based and has a maximum tool-call budget; it is not an open-ended autonomous loop.
- **Local default:** SQLite + FAISS are the default. The storage/retrieval boundaries can later be replaced with PostgreSQL + pgvector.
- **Data minimization:** only requested filings are downloaded; raw filings, processed chunks and indexes are runtime artifacts excluded from Git.

# SEC Filing Research & Financial Intelligence Copilot

A production-style AI/ML portfolio project for citation-grounded research over SEC 10-K and 10-Q filings. The system combines structure-aware filing ingestion, hybrid BM25+dense retrieval, Reciprocal Rank Fusion, cross-encoder reranking, SEC XBRL Company Facts, deterministic financial calculations, a bounded query router, source-linked citations, quantitative evaluation and local observability.

> This is not a generic chatbot. It is an evidence-constrained financial research system designed around SEC provenance.

## Overview

Given a ticker, filing type and question, the application can:

1. resolve ticker → CIK using the SEC's official ticker mapping;
2. identify the requested 10-K/10-Q and its accession metadata;
3. download only the required filing from SEC EDGAR;
4. parse headings, paragraphs and tables;
5. perform section-aware chunking;
6. build BM25 and FAISS retrieval indexes;
7. fuse sparse+dense results using Reciprocal Rank Fusion;
8. rerank passages with a cross encoder;
9. retrieve structured XBRL facts when the question is financial;
10. perform arithmetic deterministically in Python;
11. generate an evidence-constrained answer when generation is needed;
12. return SEC source citations, KPI data, retrieved passages and latency/token/cost telemetry.

## Problem

SEC filings are authoritative but long, repetitive and difficult to query reliably. Generic RAG systems often fail on financial research because they mix narrative and structured facts, perform arithmetic in the LLM, lose source provenance or present plausible but unsupported numbers.

This project separates those responsibilities:

- filing narrative → retrieval + citations;
- structured financial facts → SEC XBRL;
- calculations → deterministic Python;
- generation → only after evidence has been assembled.

## Architecture

```text
User
  ↓
Streamlit UI
  ↓
FastAPI
  ↓
Query Router
  ├── Filing Retrieval Tool
  ├── XBRL Company Facts Tool
  ├── Financial Calculation Tool
  └── RAG Retrieval Pipeline
        ↓
      BM25 + Dense FAISS
        ↓
      Reciprocal Rank Fusion
        ↓
      Cross-Encoder Reranker
        ↓
      Context Builder
        ↓
   LLM / Mock Provider
        ↓
Structured Answer + Citations + KPIs
        ↓
Evaluation + SQLite Observability
```

See [`docs/architecture.md`](docs/architecture.md) for design details.

## Features

- Official SEC EDGAR data sources
- SEC-compliant configurable User-Agent
- fair-access throttling, retry/backoff and local request cache
- ticker → CIK resolution
- latest or exact-date 10-K/10-Q retrieval
- filing metadata and source URL preservation
- HTML parsing with headings, paragraphs and tables
- structure-aware 500–900-token-style chunking with configurable overlap
- BM25 sparse retrieval
- SentenceTransformers embeddings
- FAISS dense retrieval
- Reciprocal Rank Fusion hybrid retrieval
- cross-encoder reranking
- SEC XBRL Company Facts tools with concept fallback
- revenue, net income, operating income, gross profit, assets, liabilities, cash, operating cash flow, EPS and shares outstanding support where facts are available
- deterministic YoY growth, margin and difference calculations
- bounded query routing for filing text, facts, comparisons, calculations and mixed questions
- citation IDs linked to real SEC filing URLs
- Pydantic answer schemas
- mock/demo provider plus OpenAI-compatible provider abstraction
- SQLite query/experiment metadata
- retrieval, generation, citation, financial and latency evaluation
- Streamlit research dashboard
- FastAPI endpoints
- Docker Compose
- pytest + ruff CI
- no live SEC dependency in unit tests

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11–3.14 compatible source |
| API | FastAPI, Pydantic, Uvicorn |
| UI | Streamlit |
| SEC | requests, BeautifulSoup4, lxml |
| Dense retrieval | SentenceTransformers, FAISS, NumPy |
| Sparse retrieval | rank-bm25 |
| Reranking | SentenceTransformers CrossEncoder |
| LLM | OpenAI-compatible HTTP API or local mock provider |
| Metadata | SQLite |
| Evaluation | custom metrics with RAGAS-compatible generation metric shape |
| Testing | pytest |
| Linting | ruff |
| Packaging | pyproject.toml / hatchling |
| Container | Docker, Docker Compose |

## Data Sources

Primary sources are official SEC endpoints:

- `https://www.sec.gov/files/company_tickers.json`
- `https://data.sec.gov/submissions/CIK##########.json`
- `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
- `https://data.sec.gov/api/xbrl/companyconcept/CIK##########/{taxonomy}/{concept}.json`
- `https://www.sec.gov/Archives/edgar/data/...`

**Data is fetched from SEC EDGAR at runtime and is intentionally excluded from GitHub because filings and generated indexes can be large and should not be versioned.**

Downloaded filings, processed chunks, caches, FAISS indexes, embeddings and SQLite databases are ignored by Git.

## Why SEC EDGAR?

SEC EDGAR provides first-party filing documents, submission metadata and structured XBRL facts. Using SEC sources avoids relying on scraped third-party financial pages for filing provenance, ticker resolution or core financial ground truth.

## Retrieval Pipeline

The ingestion pipeline parses the requested filing into ordered document blocks. Chunking respects section changes and paragraph/table boundaries. Every chunk preserves:

```json
{
  "chunk_id": "...",
  "ticker": "AAPL",
  "cik": "0000320193",
  "form": "10-K",
  "filing_date": "...",
  "report_date": "...",
  "section": "Item 1A. Risk Factors",
  "accession_number": "...",
  "source_url": "https://www.sec.gov/...",
  "text": "..."
}
```

## Hybrid Retrieval

Two independent rankings are built:

1. BM25 over tokenized filing chunks;
2. cosine-equivalent dense search over normalized SentenceTransformer embeddings in FAISS.

They are merged using Reciprocal Rank Fusion:

```text
RRF score = Σ 1 / (k + rank)
```

The default hybrid candidate set is 20 passages.

## Reranking

The default reranker is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

It scores query/passage pairs and keeps the top six passages by default. Reranking can be disabled through configuration for latency/cost experiments.

## XBRL Financial Tools

`CompanyFactsService` uses SEC Company Facts and concept fallback logic. Example revenue candidates include:

- `RevenueFromContractWithCustomerExcludingAssessedTax`
- `Revenues`
- `SalesRevenueNet`

Net income candidates include:

- `NetIncomeLoss`
- `ProfitLoss`

The service filters by filing form and fiscal period, normalizes the returned fact shape and preserves the actual taxonomy/concept used.

## Agent Architecture

The router classifies questions into:

- `filing_text_question`
- `financial_fact_question`
- `comparison_question`
- `calculation_question`
- `mixed_question`

It is intentionally bounded. It does **not** run an uncontrolled autonomous loop. The default maximum tool-call count is five.

Examples:

- “What risks does Apple mention?” → filing RAG
- “What was Apple's revenue in the latest fiscal year?” → XBRL only
- “Compare the latest two years of revenue.” → XBRL + deterministic calculator
- “Explain why revenue changed and calculate the growth.” → filing RAG + XBRL + calculator

When structured XBRL data can answer a financial question by itself, the application avoids an LLM call.

## Citation Design

Each retrieved passage retains filing provenance and is assigned a display citation ID such as `[1]`.

A citation includes:

- chunk ID
- SEC source URL
- section
- accession number
- form
- filing date
- evidence snippet

The generator is instructed to cite factual claims and never invent source IDs. Source validity evaluation requires citation URLs to use approved SEC hosts.

## Evaluation

Run:

```powershell
python scripts/run_evaluation.py
```

Metrics include:

### Retrieval

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- MRR
- NDCG@10

### Generation

- answer faithfulness
- context precision
- context recall
- answer relevance

### Citation

- citation correctness
- citation completeness
- citation source validity

### Financial extraction

- exact match
- absolute error
- percentage error

### System

- average latency
- P50 latency
- P95 latency
- estimated cost/query when pricing is configured

For financial evaluation cases, ground truth is fetched dynamically from SEC XBRL. Static fake financial answers are not stored in the evaluation dataset.

## Cost Optimization

The local implementation minimizes repeated work by combining:

- SEC HTTP/file caching;
- persistent filing-level FAISS indexes as an embedding cache;
- query-result caching keyed by filing, question, model and retrieval configuration;
- retrieval before generation;
- deterministic XBRL answers when no LLM is needed;
- deterministic Python arithmetic;
- a small configurable embedding model;
- optional reranking;
- provider-independent generation configuration.

## Optional PostgreSQL / pgvector Path

FAISS + SQLite are the default local stack. `VectorIndex` and `QueryRunStore` define lightweight storage boundaries, and the `postgres` optional dependency group includes `psycopg` and `pgvector` for a future server-backed implementation without making PostgreSQL mandatory.

## Security

Implemented controls include:

- environment-variable secrets
- `.env` exclusion from Git
- Pydantic input validation
- request body size limits
- maximum query length
- safe HTML parsing without script execution
- SEC-host URL allowlisting
- HTTPS-only download policy
- SSRF protection against arbitrary user URLs
- filing prompt-injection boundary
- bounded tool calls
- structured logs that do not contain API keys

Filing text is explicitly treated as untrusted data. Text such as “ignore previous instructions” inside a filing cannot override application/system instructions.

See [`docs/security.md`](docs/security.md).

## Installation

### Windows Setup

Open PowerShell in VS Code:

```powershell
git clone https://github.com/Alif1642/sec-filings-rag-intelligence.git
cd sec-filings-rag-intelligence

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"

copy .env.example .env
```

Edit `.env` and set a real SEC User-Agent containing your name/application and contact email:

```env
SEC_USER_AGENT="YourName your-email@example.com"
```

### Python compatibility

The source targets Python 3.11–3.14 syntax/API compatibility. For the most predictable local ML installation, use a Python version for which your platform currently has wheels for PyTorch, FAISS and SentenceTransformers. The Docker image uses Python 3.12.

## Environment Variables

Important variables:

```env
SEC_USER_AGENT="YourName your-email@example.com"
SEC_REQUESTS_PER_SECOND=8

DEMO_MODE=true
LLM_PROVIDER=mock

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RETRIEVAL_TOP_K=20
RERANK_TOP_K=6
RERANKER_ENABLED=true

CHUNK_TARGET_TOKENS=700
CHUNK_OVERLAP_TOKENS=80
```

For an OpenAI-compatible provider:

```env
DEMO_MODE=false
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-key
LLM_MODEL=your-model
```

Do not commit `.env`.

## Running the Project

### Ingest AAPL 10-K

```powershell
python -m ingestion.pipeline --ticker AAPL --form 10-K
```

Equivalent helper script:

```powershell
python scripts/ingest_company.py --ticker AAPL --form 10-K
```

### Build Index

```powershell
python scripts/build_index.py
```

Or point at a specific processed file:

```powershell
python scripts/build_index.py --chunks data/processed/AAPL/<accession>/chunks.json
```

### Start FastAPI

```powershell
python -m uvicorn api.main:app --reload
```

Open API docs at `http://127.0.0.1:8000/docs`.

### Start Streamlit

In a second PowerShell terminal with the same virtual environment:

```powershell
streamlit run app/streamlit_app.py
```

## Demo Mode

Demo mode is enabled by default:

```env
DEMO_MODE=true
```

Demo mode:

- still uses real SEC data when internet access is available;
- still uses local embeddings and retrieval;
- still uses deterministic XBRL financial tools/calculations;
- uses a non-fabricating mock generation provider when no paid LLM is configured;
- clearly labels mock-generated narrative as demo output.

It never creates fake financial values.

## Example Queries

```text
What was Apple's total revenue in the latest fiscal year?

Compare Apple's revenue between the latest two fiscal years.

What are Apple's major risk factors?

What does Apple say about its services business?

How did net income change year over year?

What percentage did revenue grow?

Compare revenue, net income and operating income across the last two 10-K filings.
```

## API Documentation

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | service health |
| GET | `/companies/{ticker}` | SEC ticker/CIK/company lookup |
| GET | `/companies/{ticker}/filings` | recent filing metadata |
| POST | `/ingest` | download and process requested filing |
| POST | `/query` | routed filing/financial research |
| POST | `/financials/query` | direct XBRL metric lookup |
| POST | `/compare` | latest-two-period deterministic comparison |
| POST | `/evaluate` | run evaluation dataset |
| GET | `/metrics` | local query latency metrics |

Example `/query` request:

```json
{
  "ticker": "AAPL",
  "form": "10-K",
  "question": "What were Apple's total revenues in the latest fiscal year?"
}
```

Example response shape:

```json
{
  "answer": "...",
  "citations": [],
  "kpis": [],
  "retrieved_passages": [],
  "latency_ms": 0,
  "timings_ms": {},
  "token_usage": {},
  "estimated_cost": null,
  "route": "financial_fact_question",
  "caveats": [],
  "demo_mode": true
}
```

## Docker

Create `.env` first, then:

```powershell
docker compose up --build
```

Services:

- FastAPI: `http://127.0.0.1:8000`
- Streamlit: `http://127.0.0.1:8501`

Runtime data is persisted through the local `./data` volume and remains excluded from Git.

## Testing

```powershell
python -m pytest -v
```

Lint:

```powershell
ruff check .
```

Unit tests mock SEC responses and do not require live SEC network access. Live integration checks should be run explicitly rather than as a mandatory CI dependency.

## Evaluation Results

This repository intentionally does not commit invented benchmark scores. Run the evaluation harness against current SEC data:

```powershell
python scripts/run_evaluation.py
```

The returned JSON contains per-case metrics plus average/P50/P95 latency. Record dated results in a release, experiment log or portfolio screenshot after running the benchmark in your environment.

## Limitations

- SEC filing HTML varies substantially across issuers and filing eras; table extraction is best-effort.
- Company Facts concepts differ between issuers; concept fallback improves coverage but cannot guarantee every metric.
- Cross-encoder and embedding model first-run downloads can be large.
- The default mock provider is intentionally limited; it is for software verification, not polished financial prose.
- Citation correctness validates application provenance and IDs; semantic entailment evaluation can be strengthened with a model-based judge.
- Local FAISS/SQLite are designed for single-machine portfolio use rather than multi-tenant enterprise scale.

## Future Improvements

- PostgreSQL + pgvector implementation behind the existing storage boundary
- persistent embedding/index registry with incremental updates
- richer XBRL dimensional/context resolution
- table-aware financial statement retrieval
- filing-to-filing section diffing
- model-based citation entailment judge
- RAGAS integration as an optional evaluation backend
- async SEC download workers with shared rate limiting
- OpenTelemetry tracing
- authentication and per-user quotas for deployed versions

## Project Structure

```text
sec-filings-rag-intelligence/
├── app/
│   └── streamlit_app.py
├── api/
│   ├── main.py
│   ├── dependencies.py
│   └── routes/
│       ├── company.py
│       ├── filings.py
│       ├── health.py
│       ├── metrics.py
│       └── query.py
├── ingestion/
│   ├── sec_client.py
│   ├── company_resolver.py
│   ├── filing_downloader.py
│   ├── filing_parser.py
│   ├── document_cleaner.py
│   ├── section_detector.py
│   ├── chunker.py
│   └── pipeline.py
├── src/
│   ├── config.py
│   ├── retrieval/
│   ├── rag/
│   ├── agents/
│   ├── financials/
│   ├── models/
│   └── observability/
├── evals/
│   ├── datasets/sample_questions.json
│   ├── retrieval_eval.py
│   ├── generation_eval.py
│   ├── citation_eval.py
│   ├── financial_eval.py
│   └── run_eval.py
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   ├── indexes/.gitkeep
│   └── metadata/.gitkeep
├── tests/
├── scripts/
├── docs/
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

## Resume Description

- **Engineered a citation-grounded SEC filings RAG system combining hybrid BM25+dense retrieval, cross-encoder reranking, XBRL financial-data tools, and deterministic calculations across 10-K/10-Q filings.**
- **Built an evaluation harness measuring retrieval Recall@K, MRR/NDCG, citation correctness, faithfulness, financial extraction accuracy, latency, and estimated inference cost.**

## LinkedIn Description

Built an end-to-end SEC filing research and financial intelligence copilot that resolves public-company filings directly from EDGAR, parses and structure-chunks 10-K/10-Q documents, retrieves evidence with hybrid BM25 + FAISS search, reranks with a cross encoder, and grounds financial answers in SEC XBRL Company Facts. The system separates narrative RAG from deterministic financial calculations, preserves citation provenance to original SEC filings, exposes FastAPI and Streamlit interfaces, and includes evaluation, observability, Docker and CI for a production-style AI/ML engineering portfolio project.

## GitHub Push Commands

```powershell
git init
git add .
git commit -m "Initial SEC filing RAG implementation"
git branch -M main
git remote add origin https://github.com/Alif1642/sec-filings-rag-intelligence.git
git push -u origin main
```

Before pushing, verify `.env`, downloaded SEC filings, generated indexes/embeddings and local databases are not staged.

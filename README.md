# SEC Filings RAG Intelligence

> A citation-grounded financial research platform that combines SEC EDGAR filing retrieval, structure-aware document processing, hybrid BM25 + FAISS search, cross-encoder reranking, SEC XBRL Company Facts, deterministic financial calculations, FastAPI, and Streamlit.

[![CI](https://github.com/Alif1642/sec-filings-rag-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Alif1642/sec-filings-rag-intelligence/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.14-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Research_UI-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-4B8BBE)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-yellow)
![SEC EDGAR](https://img.shields.io/badge/Data-SEC_EDGAR-1f4e79)
![XBRL](https://img.shields.io/badge/Financial_Data-XBRL-6A5ACD)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Repository:** [github.com/Alif1642/sec-filings-rag-intelligence](https://github.com/Alif1642/sec-filings-rag-intelligence)

---

## One-Minute Project Summary

**SEC Filings RAG Intelligence** is an engineered AI/NLP and financial-data system for researching public-company SEC filings. Instead of sending a question directly to an LLM, it first retrieves authoritative filing evidence and, when the question is financial, uses structured SEC XBRL Company Facts and deterministic Python calculations.

The system currently supports **10-K and 10-Q** workflows. It can resolve a ticker to a CIK, locate and download the requested filing from SEC EDGAR, parse and chunk the filing by section, build sparse and dense indexes, retrieve with **BM25 + FAISS**, fuse rankings with **Reciprocal Rank Fusion**, rerank candidates with a **cross-encoder**, and return citation-backed answers. Financial questions can bypass generation entirely and return normalized XBRL facts or deterministic comparisons.

This project is designed as a portfolio-grade example of **RAG engineering, information retrieval, financial data processing, backend/API design, evaluation, observability, testing, and containerization**—not as a notebook-only chatbot.

> **Important:** This project is a research tool and is **not investment advice**.

---

## Table of Contents

- [Why SEC Filings RAG Intelligence?](#why-sec-filings-rag-intelligence)
- [Business Value](#business-value)
- [What the System Does](#what-the-system-does)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [SEC EDGAR Data Pipeline](#sec-edgar-data-pipeline)
- [XBRL Financial Intelligence](#xbrl-financial-intelligence)
- [Retrieval-Augmented Generation Pipeline](#retrieval-augmented-generation-pipeline)
- [Document Processing](#document-processing)
- [Semantic Retrieval](#semantic-retrieval)
- [LLM-Powered Financial Question Answering](#llm-powered-financial-question-answering)
- [Financial Intelligence](#financial-intelligence)
- [API](#api)
- [Interactive Research Dashboard](#interactive-research-dashboard)
- [Data Sources](#data-sources)
- [Data Download & Runtime Artifacts](#data-download--runtime-artifacts)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [SEC User-Agent](#sec-user-agent)
- [Running the Project](#running-the-project)
- [API Documentation](#api-documentation)
- [Example Research Questions](#example-research-questions)
- [Answer & Citation Model](#answer--citation-model)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Docker](#docker)
- [Security & Responsible Use](#security--responsible-use)
- [Limitations](#limitations)
- [Future Roadmap](#future-roadmap)
- [Engineering Highlights](#engineering-highlights)
- [Why This Is More Than a Chatbot](#why-this-is-more-than-a-chatbot)
- [Demo](#demo)
- [Screenshots](#screenshots)
- [Author](#author)
- [Skills Demonstrated](#skills-demonstrated)

---

## Why SEC Filings RAG Intelligence?

SEC filings are authoritative, but they are also long, repetitive, highly structured, and information-dense. Important disclosures may be spread across sections such as **Business**, **Risk Factors**, **Management's Discussion and Analysis**, and the **Financial Statements**.

A useful research system must handle more than keyword search:

- narrative disclosures require contextual retrieval;
- financial values should come from structured financial facts when possible;
- comparisons require period-aware logic;
- arithmetic should be deterministic rather than improvised by an LLM;
- answers should preserve filing provenance and source links;
- retrieval quality matters because a fluent answer is not useful if its evidence is wrong.

This project separates those responsibilities. Narrative questions use RAG over SEC filing text, while financial fact and comparison questions use SEC XBRL Company Facts and deterministic Python calculations.

---

## Business Value

The implemented capabilities are relevant to workflows such as:

| Use case | How the system helps |
|---|---|
| Financial research | Search and question-answer over SEC 10-K/10-Q filings |
| Risk analysis | Retrieve filing sections and return source-linked evidence |
| Filing analysis | Navigate large filings through semantic and lexical retrieval |
| Financial fact lookup | Retrieve normalized facts from SEC XBRL Company Facts |
| Period comparison | Compare supported metrics using period-aware deterministic logic |
| Analyst productivity | Reduce manual navigation across large SEC documents |
| Regulatory research | Preserve filing date, accession number, section, and SEC source URL |
| Research prototyping | Expose the same capabilities through FastAPI and Streamlit |

The project does **not** provide investment recommendations or automated trading decisions.

---

## What the System Does

```mermaid
flowchart TD
    A[User] --> B[Streamlit UI or FastAPI]
    B --> C[Bounded Query Router]

    C -->|Filing text| D[SEC Filing Retrieval]
    C -->|Financial fact| E[SEC XBRL Company Facts]
    C -->|Comparison / calculation| F[Financial Comparison Logic]
    C -->|Mixed question| D
    C -->|Mixed question| E

    D --> G[SEC EDGAR Filing]
    G --> H[HTML Parsing & Cleaning]
    H --> I[Section-Aware Chunking]
    I --> J[BM25 Retrieval]
    I --> K[SentenceTransformer Embeddings + FAISS]
    J --> L[Reciprocal Rank Fusion]
    K --> L
    L --> M[Cross-Encoder Reranking]
    M --> N[Context Builder]

    E --> O[Normalized XBRL Facts]
    O --> F
    F --> P[Deterministic KPIs / Comparisons]

    N --> Q[LLM Provider or Mock Provider]
    P --> Q
    N --> R[Citation Metadata]

    Q --> S[Grounded Research Answer]
    R --> S
    P --> S

    S --> T[FastAPI Response]
    T --> U[Streamlit Dashboard]
    T --> V[SQLite Query Telemetry]
```

---

## System Architecture

The project uses a deliberately bounded architecture rather than an open-ended autonomous agent loop.

```mermaid
flowchart LR
    subgraph Sources["Authoritative SEC Sources"]
        A1[Company Ticker Mapping]
        A2[Company Submissions]
        A3[EDGAR Filing Documents]
        A4[XBRL Company Facts]
        A5[XBRL Company Concept]
    end

    subgraph Ingestion["Ingestion & Processing"]
        B1[SECClient]
        B2[FilingDownloader]
        B3[FilingParser]
        B4[StructureAwareChunker]
        B5[Processed Chunk JSON]
    end

    subgraph Retrieval["Retrieval Layer"]
        C1[BM25Retriever]
        C2[BAAI/bge-small-en-v1.5]
        C3[FAISS IndexFlatIP]
        C4[Reciprocal Rank Fusion]
        C5[cross-encoder/ms-marco-MiniLM-L-6-v2]
    end

    subgraph Financial["Structured Financial Layer"]
        D1[CompanyFactsService]
        D2[Concept / Taxonomy Fallback]
        D3[Period Selection]
        D4[FinancialComparisonService]
        D5[Deterministic Calculations]
    end

    subgraph Orchestration["Research Orchestration"]
        E1[Rule-Based Query Router]
        E2[ResearchAgent]
        E3[Context Builder]
        E4[Answer Generator]
        E5[Citation Validation]
    end

    subgraph Interfaces["Interfaces & Operations"]
        F1[FastAPI]
        F2[Streamlit]
        F3[SQLite Query Runs]
        F4[JSON Logging]
        F5[Evaluation Suite]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B2
    B1 --> B2
    B2 --> B3 --> B4 --> B5
    B5 --> C1
    B5 --> C2 --> C3
    C1 --> C4
    C3 --> C4 --> C5

    A4 --> D1
    A5 --> D1
    D1 --> D2 --> D3 --> D4 --> D5

    C5 --> E3
    D1 --> E2
    D4 --> E2
    E1 --> E2
    E2 --> E3 --> E4 --> E5

    E5 --> F1
    F1 --> F2
    F1 --> F3
    F1 --> F4
    E2 --> F5
```

### Architectural principles

- **Deterministic first:** structured financial facts and arithmetic do not require an LLM.
- **Evidence first:** narrative generation happens only after filing evidence is retrieved.
- **Bounded routing:** the router is rule-based and constrained by a configurable tool-call budget.
- **Source provenance:** filing citations retain section and SEC URL metadata.
- **Local-first runtime:** FAISS and SQLite are the default local storage technologies.
- **Replaceable boundaries:** interfaces are structured so components such as vector or metadata stores can evolve without redesigning the entire application.

---

## Core Features

| Capability | Implementation |
|---|---|
| SEC ticker resolution | Official SEC company ticker mapping |
| CIK normalization | Zero-padded 10-digit CIK handling |
| Filing discovery | SEC submissions metadata |
| Supported filing types | 10-K and 10-Q |
| Filing download | Requested filing only, cached locally |
| HTML processing | BeautifulSoup + lxml |
| Section extraction | SEC item-heading detection |
| Table extraction | Tables normalized into compact text rows |
| Chunking | Section-aware chunking with bounded overlap |
| Sparse retrieval | BM25 |
| Dense retrieval | SentenceTransformers + FAISS |
| Embedding model | `BAAI/bge-small-en-v1.5` by default |
| Hybrid fusion | Reciprocal Rank Fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` by default |
| RAG context | Bounded context builder with stable citation IDs |
| LLM layer | Mock/demo provider or OpenAI-compatible `/chat/completions` provider |
| Financial facts | SEC XBRL Company Facts |
| Financial comparisons | Period-aware deterministic Python logic |
| Query routing | Filing text, financial fact, comparison, calculation, mixed |
| API | FastAPI |
| UI | Streamlit |
| Metadata / telemetry | SQLite query-run store |
| Logging | Structured JSON logging |
| Evaluation | Retrieval, generation-grounding, citation, financial, latency |
| Testing | pytest |
| CI | GitHub Actions: ruff, pytest, API smoke test |
| Containerization | Docker + Docker Compose |

---

## SEC EDGAR Data Pipeline

### 1. Company identification

A ticker is validated, normalized to uppercase, and resolved against the SEC's official company ticker mapping.

```text
Ticker
  ↓
SEC company_tickers.json
  ↓
CIK + Company Name
```

### 2. CIK handling

CIKs are normalized to the SEC's 10-digit zero-padded format.

```text
320193 → 0000320193
```

### 3. Filing discovery

The SEC submissions endpoint is used to locate recent filing metadata. The application supports:

- `10-K`
- `10-Q`

A specific filing date can be supplied; otherwise the latest matching filing is selected.

### 4. Filing metadata

The pipeline preserves metadata including:

- ticker;
- CIK;
- form;
- filing date;
- report date;
- accession number;
- primary document;
- SEC filing URL.

### 5. Filing download

`FilingDownloader` downloads only the requested filing and stores the raw document under the runtime data directory.

### 6. Parsing

`FilingParser` converts SEC HTML into ordered blocks:

- headings;
- paragraphs;
- compact table text.

Inline-XBRL metadata containers such as `ix:header`, `ix:hidden`, `ix:references`, and `ix:resources` are dropped from the text parsing path.

### 7. Section detection

Detected SEC item headings are preserved as section metadata so retrieval can reason over filing structure instead of treating the document as one flat text blob.

### 8. Chunking

`StructureAwareChunker`:

- targets configurable chunk sizes;
- does not carry overlap across section boundaries;
- uses bounded overlap within a section;
- splits oversized text blocks;
- removes short table-of-contents fragments while preserving legitimate short sections.

### 9. Storage and indexing

Processed chunks are written to:

```text
data/processed/<TICKER>/<ACCESSION>/chunks.json
```

FAISS indexes and their associated chunk metadata are persisted under:

```text
data/indexes/<FILING_KEY>/
```

---

## XBRL Financial Intelligence

SEC filing text and XBRL Company Facts solve different problems:

| Unstructured filing text | Structured XBRL facts |
|---|---|
| Business explanations | Numeric financial facts |
| Risk disclosures | Revenue |
| Management commentary | Net income |
| Strategy and competition | Operating income |
| Drivers of period changes | Gross profit |
| Narrative context | Assets / liabilities |
| Retrieved through RAG | Retrieved through SEC XBRL APIs |

### Supported normalized financial metrics

The current metric mapping supports:

- `revenue`
- `net_income`
- `operating_income`
- `gross_profit`
- `assets`
- `liabilities`
- `cash`
- `operating_cash_flow`
- `eps`
- `shares_outstanding`

The financial layer uses concept-priority and taxonomy fallback logic over `us-gaap` and `dei` where applicable.

### Period-aware handling

Financial facts are normalized differently for annual and quarterly reports:

- **10-K duration metrics** prefer annual contexts.
- **10-Q revenue/income/EPS-style metrics** prefer discrete quarterly contexts when available.
- **10-Q cumulative cash-flow facts** can retain YTD contexts when a discrete quarter does not exist.
- Quarterly YTD comparisons attempt to compare the same fiscal period from the prior fiscal year rather than incorrectly comparing different-duration periods.

This matters because a financial RAG system should not treat all XBRL contexts as interchangeable.

---

## Retrieval-Augmented Generation Pipeline

```text
Question
  ↓
Query routing
  ↓
Filing retrieval / structured financial lookup
  ↓
BM25 + dense semantic retrieval
  ↓
Reciprocal Rank Fusion
  ↓
Cross-encoder reranking
  ↓
Bounded context construction
  ↓
LLM/mock generation when required
  ↓
Citation validation
  ↓
Answer + sources + KPIs + telemetry
```

### Why RAG?

A language model alone does not know which filing or filing section should be treated as the authoritative source for a question. The RAG path supplies retrieved SEC evidence before generation.

The design aims to improve:

- **grounding** — responses are based on retrieved filing passages;
- **traceability** — evidence carries SEC source URLs and filing metadata;
- **domain relevance** — retrieval is performed over the requested filing;
- **hallucination resistance** — the generator is explicitly constrained to supplied evidence;
- **separation of concerns** — structured financial facts are handled separately from narrative filing text.

RAG reduces hallucination risk; it does not eliminate it.

---

## Document Processing

```mermaid
flowchart LR
    A[Raw SEC HTML] --> B[Drop executable / irrelevant tags]
    B --> C[Extract headings, paragraphs, tables]
    C --> D[Clean text]
    D --> E[Detect SEC sections]
    E --> F[Flag instruction-like text]
    F --> G[Structure-aware chunking]
    G --> H[Attach filing metadata]
    H --> I[Processed chunks]
    I --> J[BM25 + FAISS indexing]
```

Each chunk stores fields such as:

```json
{
  "chunk_id": "...",
  "ticker": "...",
  "cik": "...",
  "form": "10-K",
  "filing_date": "...",
  "report_date": "...",
  "accession_number": "...",
  "section": "...",
  "source_url": "https://www.sec.gov/...",
  "text": "...",
  "anchor": null,
  "untrusted_instruction_like": false
}
```

The `untrusted_instruction_like` flag is part of the defensive design for treating filing content as data rather than executable model instructions.

---

## Semantic Retrieval

### Sparse retrieval — BM25

`BM25Retriever` tokenizes filing chunks and ranks them lexically. The implementation uses `rank-bm25` when available and includes a small fallback BM25 implementation for minimal environments.

### Dense retrieval — SentenceTransformers + FAISS

The default embedding model is:

```text
BAAI/bge-small-en-v1.5
```

Embeddings are normalized and indexed with:

```text
faiss.IndexFlatIP
```

With normalized embeddings, inner-product search acts as cosine-equivalent similarity search.

### Hybrid fusion

Sparse and dense rankings are merged using Reciprocal Rank Fusion:

```text
RRF score = Σ 1 / (k + rank)
```

The configured retrieval candidate count defaults to `20`.

### Cross-encoder reranking

The default reranker is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

It scores query-passage pairs and keeps the configured top results (`6` by default).

### Section preference

The current RAG pipeline includes a targeted preference for `Item 1A` evidence when a query clearly asks about major/business risk factors and enough Item 1A hits are already present.

---

## LLM-Powered Financial Question Answering

The generation layer is provider-based.

### Default: mock/demo provider

The repository defaults to:

```env
DEMO_MODE=true
LLM_PROVIDER=mock
```

The mock provider is intentionally conservative. It extracts relevant filing evidence and can combine that evidence with structured financial facts. It is useful for demonstrating the end-to-end architecture without requiring an external LLM API key.

### Optional: OpenAI-compatible provider

The project also implements an HTTP provider for APIs exposing an OpenAI-compatible:

```text
/chat/completions
```

interface.

Configuration is controlled by:

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
```

### Prompt strategy

The system prompt requires the generator to:

- use only supplied evidence and structured facts;
- cite filing-based claims;
- avoid fabricating SEC values, dates, citations, or source URLs;
- distinguish filing text, XBRL facts, and deterministic calculations;
- ignore instructions embedded inside filing content;
- return an explicit insufficient-evidence response when necessary.

---

## Financial Intelligence

Financial queries use a deterministic-first path.

### Supported operations

- latest metric lookup;
- fiscal-year metric lookup;
- latest-two-period comparison;
- year-over-year growth;
- absolute difference;
- margin calculation utility;
- quarterly comparison logic that accounts for duration differences;
- structured KPI output.

### Query routing

The router classifies questions into:

```text
filing_text_question
financial_fact_question
comparison_question
calculation_question
mixed_question
```

Examples:

```text
"What risks does the company disclose?"
→ filing_text_question
```

```text
"What was revenue?"
→ financial_fact_question
```

```text
"Compare revenue year over year."
→ comparison_question
```

```text
"Explain why revenue changed and calculate the growth."
→ mixed_question
```

For pure financial facts/comparisons/calculations, the application can return a deterministic answer without invoking an LLM.

---

## API

FastAPI exposes the research and ingestion capabilities as REST endpoints.

### Endpoint summary

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/companies/{ticker}` | Resolve company metadata |
| `GET` | `/companies/{ticker}/filings` | List recent filings, optionally filtered to 10-K/10-Q |
| `POST` | `/ingest` | Download, parse, chunk, and persist a filing |
| `POST` | `/query` | Run routed filing/financial research |
| `POST` | `/financials/query` | Retrieve a normalized XBRL metric |
| `POST` | `/compare` | Compare the latest two applicable periods |
| `POST` | `/evaluate` | Run the repository evaluation dataset |
| `GET` | `/metrics` | Return query-latency statistics from SQLite |

<details>
<summary><strong>GET /health</strong></summary>

**Request**

```http
GET /health
```

**Response**

```json
{
  "status": "ok"
}
```

</details>

<details>
<summary><strong>GET /companies/{ticker}</strong></summary>

**Parameters**

- `ticker` — ticker symbol in the path.

**Request**

```http
GET /companies/AAPL
```

**Response shape**

```json
{
  "ticker": "AAPL",
  "cik": "<10-digit SEC CIK>",
  "title": "<SEC company name>"
}
```

</details>

<details>
<summary><strong>GET /companies/{ticker}/filings</strong></summary>

**Parameters**

- `ticker` — path parameter.
- `form` — optional `10-K` or `10-Q`.
- `limit` — `1` to `100`, default `20`.

**Request**

```http
GET /companies/AAPL/filings?form=10-K&limit=5
```

**Response shape**

```json
{
  "company": {
    "ticker": "AAPL",
    "cik": "<CIK>",
    "title": "<company>"
  },
  "filings": [
    {
      "accession_number": "<accession>",
      "filing_date": "<date>",
      "report_date": "<date>",
      "form": "10-K",
      "primary_document": "<document>"
    }
  ]
}
```

</details>

<details>
<summary><strong>POST /ingest</strong></summary>

**Request**

```json
{
  "ticker": "AAPL",
  "form": "10-K",
  "filing_date": null
}
```

**Response shape**

```json
{
  "metadata": {},
  "raw_path": "<local runtime path>",
  "chunks_path": "<local runtime path>",
  "block_count": "<integer>",
  "chunk_count": "<integer>"
}
```

</details>

<details>
<summary><strong>POST /query</strong></summary>

**Request**

```json
{
  "ticker": "AAPL",
  "form": "10-K",
  "filing_date": null,
  "question": "What are the company's major risk factors?"
}
```

**Response shape**

```json
{
  "answer": "<grounded answer>",
  "citations": [],
  "kpis": [],
  "retrieved_passages": [],
  "latency_ms": 0.0,
  "timings_ms": {
    "retrieval": 0.0,
    "reranking": 0.0,
    "generation": 0.0,
    "total": 0.0
  },
  "token_usage": {
    "input": 0,
    "output": 0
  },
  "estimated_cost": null,
  "route": "<query route>",
  "caveats": [],
  "demo_mode": true
}
```

</details>

<details>
<summary><strong>POST /financials/query</strong></summary>

**Request**

```json
{
  "ticker": "AAPL",
  "metric": "revenue",
  "fiscal_year": null,
  "form": "10-K"
}
```

**Response shape**

```json
{
  "company": {},
  "fact": {
    "metric": "revenue",
    "value": "<SEC numeric value>",
    "unit": "USD",
    "fiscal_year": "<year>",
    "fiscal_period": "<period>",
    "form": "10-K",
    "filed": "<date>",
    "start": "<date or null>",
    "end": "<date or null>",
    "accession_number": "<accession>",
    "concept": "<XBRL concept>",
    "taxonomy": "<taxonomy>",
    "source": "SEC Company Facts"
  }
}
```

</details>

<details>
<summary><strong>POST /compare</strong></summary>

**Request**

```json
{
  "ticker": "AAPL",
  "metric": "revenue",
  "form": "10-K"
}
```

**Response shape**

```json
{
  "company": {},
  "comparison": {
    "metric": "revenue",
    "current": {},
    "previous": {},
    "difference": "<number>",
    "growth_pct": "<number>",
    "calculation_source": "Deterministic Python calculation over SEC XBRL facts"
  }
}
```

</details>

<details>
<summary><strong>POST /evaluate</strong></summary>

**Request**

```json
{
  "dataset_path": "evals/datasets/sample_questions.json"
}
```

**Response**

Returns per-question evaluation rows plus aggregate statistics produced by the repository's evaluation runner.

</details>

<details>
<summary><strong>GET /metrics</strong></summary>

**Request**

```http
GET /metrics
```

**Response shape**

```json
{
  "query_latency": {
    "count": "<integer>",
    "average_ms": "<number>",
    "p50_ms": "<number>",
    "p95_ms": "<number>"
  }
}
```

</details>

---

## Interactive Research Dashboard

The Streamlit application provides a research interface over the FastAPI backend.

### Implemented UI controls

- ticker input;
- filing type selection (`10-K`, `10-Q`);
- optional filing date;
- natural-language question;
- display of configured retrieval candidate count;
- display of reranker state;
- display of model/provider and temperature;
- demo-mode indicator.

### Implemented result panels

- generated/deterministic answer;
- caveats;
- KPI table;
- source citations;
- filing date and accession metadata;
- direct SEC filing link;
- retrieved passages;
- retrieval, reranking, generation, and total API timing;
- route;
- chunk count;
- token usage;
- estimated cost when pricing is configured.

This makes the system usable by non-technical reviewers without removing the underlying API.

---

## Data Sources

The application uses official SEC endpoints as its primary data sources:

| Source | Purpose |
|---|---|
| SEC company ticker mapping | Ticker → CIK |
| SEC submissions API | Filing metadata |
| SEC EDGAR archives | Filing documents |
| SEC XBRL Company Facts | Company-level structured financial facts |
| SEC XBRL Company Concept | Concept-specific financial data access |

The client restricts requests to:

```text
https://www.sec.gov
https://data.sec.gov
```

No third-party scraped finance site is required for the core filing or financial-fact workflow.

---

## Data Download & Runtime Artifacts

Large SEC documents and generated indexes are intentionally not committed to Git.

Data is obtained at runtime through the ingestion pipeline:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_company.py --ticker AAPL --form 10-K
```

or through the API:

```http
POST /ingest
```

Runtime data is organized under:

```text
data/
├── raw/         # downloaded filing documents
├── processed/   # normalized chunks + manifests
├── indexes/     # FAISS indexes + indexed chunk metadata
├── metadata/    # SQLite metadata / query runs
└── cache/       # SEC request cache + RAG query cache
```

The repository keeps directory placeholders while excluding generated/downloaded artifacts from version control.

---

## Project Structure

```text
sec-filings-rag-intelligence/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── __init__.py
│   ├── dependencies.py
│   ├── main.py
│   └── routes/
│       ├── __init__.py
│       ├── company.py
│       ├── filings.py
│       ├── health.py
│       ├── metrics.py
│       └── query.py
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
├── data/
│   ├── indexes/
│   │   └── .gitkeep
│   ├── metadata/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   ├── retrieval.md
│   └── security.md
├── evals/
│   ├── __init__.py
│   ├── citation_eval.py
│   ├── financial_eval.py
│   ├── generation_eval.py
│   ├── retrieval_eval.py
│   ├── run_eval.py
│   └── datasets/
│       └── sample_questions.json
├── ingestion/
│   ├── __init__.py
│   ├── chunker.py
│   ├── company_resolver.py
│   ├── document_cleaner.py
│   ├── filing_downloader.py
│   ├── filing_parser.py
│   ├── pipeline.py
│   ├── sec_client.py
│   └── section_detector.py
├── scripts/
│   ├── build_index.py
│   ├── ingest_company.py
│   └── run_evaluation.py
├── src/
│   ├── agents/
│   │   ├── calculator_tool.py
│   │   ├── filing_tool.py
│   │   ├── research_agent.py
│   │   ├── router.py
│   │   └── xbrl_tool.py
│   ├── financials/
│   │   ├── calculations.py
│   │   ├── companyfacts.py
│   │   ├── comparison.py
│   │   └── metrics.py
│   ├── models/
│   │   ├── database.py
│   │   └── schemas.py
│   ├── observability/
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── tracing.py
│   ├── rag/
│   │   ├── answer_schema.py
│   │   ├── citation.py
│   │   ├── context.py
│   │   ├── generator.py
│   │   ├── pipeline.py
│   │   └── prompts.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── fusion.py
│   │   ├── hybrid.py
│   │   ├── reranker.py
│   │   └── vector_store.py
│   ├── __init__.py
│   └── config.py
├── tests/
│   ├── test_api.py
│   ├── test_chunker.py
│   ├── test_citations.py
│   ├── test_company_resolver.py
│   ├── test_financials.py
│   ├── test_generator.py
│   ├── test_parser.py
│   ├── test_rag_section_preference.py
│   ├── test_retrieval.py
│   └── test_sec_client.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── docker-compose.yml
└── pyproject.toml
```

> This repository uses `pyproject.toml` for packaging and dependencies; it does not require a separate `requirements.txt`.

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11–3.14 | Core implementation |
| Validation / settings | Pydantic + pydantic-settings | Request and configuration validation |
| SEC access | requests + urllib3 retry | SEC EDGAR / XBRL HTTP client |
| HTML parsing | BeautifulSoup4 + lxml | Filing extraction |
| Sparse retrieval | rank-bm25 | Lexical ranking |
| Embeddings | SentenceTransformers | Semantic vectors |
| Default embedding model | BAAI/bge-small-en-v1.5 | Dense representation |
| Vector search | FAISS | Local dense retrieval |
| Rank fusion | Reciprocal Rank Fusion | Hybrid sparse+dense ranking |
| Reranking | SentenceTransformers CrossEncoder | Query/passage reranking |
| Default reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Candidate scoring |
| LLM integration | Mock provider / OpenAI-compatible HTTP | Evidence-based generation |
| Financial data | SEC XBRL Company Facts | Structured financial facts |
| Calculations | Python | Deterministic growth, difference, margin |
| API | FastAPI + Uvicorn | REST interface |
| UI | Streamlit + pandas | Interactive research dashboard |
| Metadata store | SQLite | Query-run telemetry |
| Evaluation | Custom evaluation modules | Retrieval, citation, generation, financial metrics |
| Testing | pytest | Automated tests |
| Linting | ruff | Static code quality |
| CI | GitHub Actions | Lint, tests, API smoke test |
| Packaging | hatchling / pyproject.toml | Python packaging |
| Containerization | Docker + Docker Compose | API and UI containers |

### Not currently implemented as the default runtime

`pyproject.toml` exposes optional PostgreSQL/pgvector dependencies, but the repository's implemented default metadata and vector backends are **SQLite and FAISS**. PostgreSQL/pgvector should therefore be treated as an extension point rather than a completed backend.

---

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/Alif1642/sec-filings-rag-intelligence.git
cd sec-filings-rag-intelligence

py -3.13 -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Copy-Item .env.example .env
```

Using the virtual environment's Python executable directly avoids PowerShell execution-policy issues with `Activate.ps1`.

### Configure SEC access

Edit `.env` and replace the placeholder:

```env
SEC_USER_AGENT="YourName your-email@example.com"
```

The default demo configuration does not require an external LLM API key.

---

## Environment Variables

The application configuration is defined in `src/config.py`; `.env.example` includes the main runtime settings.

| Variable | Default / example | Purpose |
|---|---|---|
| `APP_NAME` | application default | FastAPI application name |
| `ENVIRONMENT` | `development` | Environment label |
| `DATA_DIR` | `data` | Runtime data directory |
| `SEC_USER_AGENT` | placeholder | Required descriptive SEC User-Agent |
| `SEC_TIMEOUT_SECONDS` | `30` | SEC request timeout |
| `SEC_MAX_RETRIES` | `4` | SEC retry count |
| `SEC_REQUESTS_PER_SECOND` | `8` | SEC request throttle; validation limits this to `<=10` |
| `SEC_CACHE_TTL_SECONDS` | `3600` | SEC response-cache TTL |
| `DEMO_MODE` | `true` | Force mock/demo generation path |
| `LLM_PROVIDER` | `mock` | `mock` or `openai_compatible` |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Base URL for compatible LLM API |
| `LLM_API_KEY` | empty | API credential for external LLM provider |
| `LLM_MODEL` | `gpt-4.1-mini` | Model identifier sent to compatible provider |
| `LLM_TEMPERATURE` | `0` | Generation temperature |
| `LLM_TIMEOUT_SECONDS` | `60` | LLM request timeout |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | SentenceTransformer embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `RETRIEVAL_TOP_K` | `20` | Hybrid retrieval candidate count |
| `RERANK_TOP_K` | `6` | Final reranked passage count |
| `RERANKER_ENABLED` | `true` | Enable/disable reranking |
| `CHUNK_TARGET_TOKENS` | `700` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `80` | Same-section overlap budget |
| `CHUNK_MIN_TOKENS` | `120` | Threshold used by short TOC-fragment filtering |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Streamlit → API address |
| `MAX_QUERY_CHARS` | `5000` | Maximum research-question length |
| `MAX_TOOL_CALLS` | `5` | Router/orchestrator tool-call budget |
| `REQUEST_SIZE_LIMIT_BYTES` | `1000000` | FastAPI body-size limit |
| `SQLITE_PATH` | `data/metadata/app.db` | Local query telemetry database |
| `PRICING_INPUT_PER_1M` | unset | Optional input-token price |
| `PRICING_OUTPUT_PER_1M` | unset | Optional output-token price |

Never commit a real `.env` file or API credential.

---

## SEC User-Agent

Automated SEC access should use a descriptive User-Agent.

```env
SEC_USER_AGENT="YourName your-email@example.com"
```

The SEC client sends this header on requests and also implements throttling, caching, retries, and `Retry-After` support for transient failures.

The repository's `.env.example` contains a placeholder only. Keep personal contact information in your local `.env`, which is ignored by Git.

---

## Running the Project

### 1. Start the FastAPI backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 2. Start Streamlit in a second terminal

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

### 3. Ingest from the command line

```powershell
.\.venv\Scripts\python.exe scripts\ingest_company.py --ticker AAPL --form 10-K
```

### 4. Build an index

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py --ticker AAPL --form 10-K
```

---

## API Documentation

FastAPI's generated documentation is available while the API is running:

```text
Swagger UI: http://127.0.0.1:8000/docs
ReDoc:      http://127.0.0.1:8000/redoc
```

---

## Example Research Questions

These are example **questions**, not fabricated system outputs.

### Filing narrative

```text
What are the company's major risk factors?
```

```text
What does the latest 10-K say about competition?
```

```text
What does management say about liquidity?
```

### Structured financial facts

```text
What was revenue in the latest fiscal year?
```

```text
What was net income in the latest 10-K?
```

```text
What were total assets?
```

### Comparison and calculation

```text
How did revenue change year over year?
```

```text
Compare operating income across the latest two annual periods.
```

### Mixed narrative + financial reasoning

```text
Explain why revenue changed and calculate the growth.
```

---

## Answer & Citation Model

For filing-text questions, the system follows this conceptual structure:

```text
Question
  ↓
Retrieved SEC evidence
  ↓
Reranked passages
  ↓
Bounded context
  ↓
Generated answer
  ↓
Citation validation
  ↓
Section + accession + filing date + SEC URL
```

A citation object can contain:

```json
{
  "citation_id": "1",
  "chunk_id": "<chunk>",
  "source_url": "https://www.sec.gov/...",
  "section": "Item 1A. Risk Factors",
  "accession_number": "<accession>",
  "form": "10-K",
  "filing_date": "<date>",
  "snippet": "<supporting evidence>"
}
```

Citation IDs referenced by generated text are checked against the context's known citations. The evaluation layer separately checks citation correctness, completeness, and whether citation URLs point to approved SEC hosts.

---

## Evaluation

The repository contains a custom evaluation framework. It does **not** publish fabricated benchmark scores.

### Retrieval metrics

Implemented in `evals/retrieval_eval.py`:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- NDCG@10

### Generation-grounding baseline

`LexicalGroundingJudge` provides an offline, lexical baseline with metrics shaped as:

- `answer_faithfulness`
- `context_precision`
- `context_recall`
- `answer_relevance`

This is a lightweight baseline and should not be presented as equivalent to a model-based semantic judge.

### Citation evaluation

Implemented metrics:

- citation correctness;
- citation completeness;
- citation source validity.

### Financial evaluation

Implemented metrics:

- exact match;
- absolute error;
- percentage error.

### Latency

The evaluation runner records per-question latency and aggregate timing statistics.

Run the evaluation suite with:

```powershell
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

or via:

```http
POST /evaluate
```

---

## Testing

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The automated tests cover implemented behavior including:

- FastAPI health and query endpoints without live SEC dependence;
- ticker resolution and CIK normalization;
- filing parser section/paragraph/table extraction;
- structure-aware chunk metadata;
- BM25 retrieval;
- Reciprocal Rank Fusion;
- FAISS vector search with controlled test embeddings;
- citation validation and completeness;
- financial fact normalization;
- deterministic financial calculations;
- period-aware quarterly YTD comparison;
- mock generator evidence handling;
- safe insufficient-evidence behavior;
- risk-query section preference.

The CI workflow deliberately avoids requiring live SEC network access for unit tests.

### Lint

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

### GitHub Actions

On push and pull request, CI is configured to:

1. install Python 3.12;
2. install `.[dev]`;
3. run `ruff check .`;
4. run pytest;
5. launch FastAPI and verify `/health`.

---

## Docker

The repository defines two Docker Compose services:

| Service | Port | Role |
|---|---:|---|
| `api` | `8000` | FastAPI backend |
| `streamlit` | `8501` | Streamlit dashboard |

Both mount the local `./data` directory into `/app/data`.

### Start

```powershell
docker compose up --build
```

### Run detached

```powershell
docker compose up -d
```

### Inspect

```powershell
docker compose ps
```

### Stop

```powershell
docker compose down
```

The Streamlit container addresses the backend internally as:

```text
http://api:8000
```

---

## Security & Responsible Use

### Implemented controls

**Environment-based secrets**

- configuration is loaded from environment variables / `.env`;
- `.env` is excluded from version control;
- `.env.example` uses placeholders;
- API keys are not intentionally logged.

**Input validation**

- Pydantic validates ticker, filing type, question length, and request schemas;
- ticker input is normalized;
- the API rejects oversized request bodies;
- the research agent enforces a maximum question length;
- tool calls are bounded.

**SEC URL validation / SSRF reduction**

`SECClient` only allows HTTPS requests to:

```text
www.sec.gov
data.sec.gov
```

Embedded credentials and non-standard ports are rejected.

**Safe filing parsing**

Scripts, styles, SVG, `noscript`, and selected inline-XBRL metadata elements are removed before retrieval text is built. Downloaded filing HTML is parsed as data and is not executed.

**Prompt-injection awareness**

Instruction-like filing text can be flagged, and the system prompt explicitly tells the generation layer not to follow instructions embedded in filing content.

**SEC request controls**

The SEC client implements:

- configurable throttling;
- retry/backoff;
- HTTP 429 handling;
- local response caching.

### Production recommendations not yet implemented

- API authentication / authorization;
- production secret manager;
- user-level quotas;
- external rate limiting;
- network isolation;
- production-grade tracing/metrics backend;
- hardened deployment configuration.

### Responsible use

Generated research can still be incomplete or incorrect. Validate material financial conclusions against the underlying SEC filing and structured facts.

**This project is not financial or investment advice.**

---

## Limitations

The repository intentionally exposes its current boundaries.

1. **10-K and 10-Q only**  
   The ingestion/request schemas currently restrict supported filing forms to 10-K and 10-Q.

2. **Local FAISS + SQLite default**  
   FAISS and SQLite are implemented. PostgreSQL/pgvector are extension points, not completed production backends.

3. **Demo provider is heuristic**  
   The default mock provider is an evidence-extraction demo path rather than a general-purpose LLM.

4. **External LLM integration requires a compatible API**  
   Non-demo generation expects an OpenAI-compatible `/chat/completions` endpoint and configured API key.

5. **Retrieval is not perfect**  
   Hybrid retrieval and reranking improve ranking, but irrelevant or neighboring sections can still enter the candidate set. Section preference is currently heuristic.

6. **Finite XBRL concept mappings**  
   Metric normalization depends on the repository's configured concept fallbacks and may not cover every issuer-specific reporting variation.

7. **SEC filing formats vary**  
   Parsing heuristics can be affected by unusual HTML structure or issuer-specific formatting.

8. **Evaluation uses a lightweight generation baseline**  
   The lexical grounding judge is useful for offline regression checks but is not a substitute for a rigorous model-based evaluation program.

9. **No hosted production deployment is included**  
   The repository contains local and Docker execution paths but does not claim a deployed cloud service.

10. **No public-API authentication**  
    FastAPI authentication / RBAC is not currently implemented.

11. **ML dependencies make container builds heavy**  
    SentenceTransformers/PyTorch-based retrieval and reranking can make first-time installation and Docker builds large and slow.

12. **Live data depends on SEC availability**  
    Ingestion and XBRL retrieval depend on SEC endpoints, network connectivity, and fair-access constraints.

---

## Future Roadmap

These items are **future improvements**, not current features.

### Retrieval and RAG

- stronger metadata- and section-aware filtering;
- retrieval regression datasets with labeled relevant chunks;
- improved query rewriting for more filing-question classes;
- multi-filing reasoning;
- multi-company comparative research;
- streaming answer support;
- model/prompt version tracking.

### SEC and financial data

- additional forms such as 8-K, DEF 14A, and 20-F;
- incremental new-filing ingestion;
- automated filing update monitoring;
- broader XBRL concept normalization;
- richer financial-statement analytics.

### Evaluation

- model-based faithfulness/relevance judge;
- versioned evaluation baselines;
- CI regression thresholds;
- larger labeled retrieval datasets.

### Platform

- implemented PostgreSQL metadata backend;
- implemented pgvector backend;
- background ingestion jobs;
- authentication / RBAC;
- user quotas and API rate limiting;
- Prometheus/OpenTelemetry-style production telemetry;
- cloud deployment;
- container image optimization and CPU-focused ML packaging.

---

## Engineering Highlights

### 1. Financial Data Engineering

The project combines two authoritative but structurally different data sources:

- filing documents from SEC EDGAR;
- structured SEC XBRL Company Facts.

The code includes concept fallback, taxonomy handling, annual/quarterly period selection, and duration-aware comparison logic.

### 2. NLP & Information Retrieval

The retrieval stack combines:

- document cleaning;
- SEC section extraction;
- section-aware chunking;
- BM25;
- SentenceTransformer embeddings;
- FAISS;
- Reciprocal Rank Fusion;
- cross-encoder reranking.

### 3. Retrieval-Augmented Generation

Generation is downstream of evidence retrieval. The model receives bounded evidence and structured facts rather than an unconstrained prompt.

### 4. Backend Engineering

FastAPI exposes ingestion, company lookup, filing lookup, research, structured financial lookup, comparisons, evaluation, and metrics.

### 5. AI Application Engineering

The architecture separates:

```text
retrieval
financial facts
calculation
context
generation
citations
evaluation
```

instead of hiding all behavior inside one LLM call.

### 6. Structured + Unstructured Data Architecture

Narrative evidence and financial facts are intentionally processed through different paths and recombined only when a mixed question requires both.

### 7. Explainability & Traceability

Answer responses can return:

- citations;
- SEC URLs;
- filing sections;
- accession numbers;
- filing dates;
- retrieved passages;
- KPIs;
- route and latency metadata.

### 8. Testing

Unit tests isolate network-dependent behavior and validate critical retrieval, parsing, financial, citation, and API logic.

### 9. Deployment

The same repository can run locally or through Docker Compose with separate API and Streamlit services.

### 10. Production-Oriented Design

The codebase includes:

- environment-based configuration;
- input validation;
- bounded tool use;
- cache invalidation versioning for RAG queries;
- SEC host validation;
- request-size control;
- structured logging;
- query telemetry;
- deterministic financial calculations;
- CI checks.

---

## Why This Is More Than a Chatbot

A traditional chatbot is approximately:

```text
User
  ↓
LLM
  ↓
Answer
```

This project is:

```text
SEC EDGAR
  ↓
Filing discovery
  ↓
Filing ingestion
  ↓
HTML parsing / cleaning
  ↓
Section-aware chunking
  ↓
BM25 + embeddings + FAISS
  ↓
Reciprocal Rank Fusion
  ↓
Cross-encoder reranking
  ↓
Retrieved filing evidence
  ┐
  ├──→ Query orchestration → Grounded answer → Citations → API → Dashboard
  │
SEC XBRL Company Facts
  ↓
Period normalization
  ↓
Deterministic financial calculations
  ┘
```

The technically important part is not the chat interface. It is the **data, retrieval, financial reasoning, provenance, orchestration, evaluation, and software-engineering pipeline behind the interface**.

---

## Demo

A hosted live demo is not currently claimed.

### Local demo

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

Or start both services with Docker:

```powershell
docker compose up --build
```

---

## Screenshots

No screenshots are fabricated in this README. Recommended portfolio captures:

| Screenshot | Recommended path | Status |
|---|---|---|
| Streamlit dashboard overview | `docs/screenshots/streamlit-dashboard.png` | TODO |
| Filing-text RAG answer | `docs/screenshots/rag-answer.png` | TODO |
| Citation / SEC source cards | `docs/screenshots/citations.png` | TODO |
| KPI / XBRL result | `docs/screenshots/xbrl-financials.png` | TODO |
| Retrieved evidence expanders | `docs/screenshots/retrieved-evidence.png` | TODO |
| Performance metrics panel | `docs/screenshots/performance.png` | TODO |
| FastAPI Swagger UI | `docs/screenshots/swagger.png` | TODO |
| Architecture diagram capture | `docs/screenshots/architecture.png` | TODO |

A strong portfolio version should place the Streamlit dashboard screenshot near the top of this README once real screenshots are available.

---

## Author

**Md. Alif Hossen**  
B.Sc. in Computer Science & Engineering  
Daffodil International University

- GitHub: [Alif1642](https://github.com/Alif1642)
- LinkedIn: [md-alif-hossen1642](https://www.linkedin.com/in/md-alif-hossen1642/)

---

## Skills Demonstrated

### AI / ML

- Retrieval-Augmented Generation
- NLP
- LLM integration
- Sentence embeddings
- semantic retrieval
- sparse retrieval
- hybrid retrieval
- cross-encoder reranking
- grounded answer generation

### Financial Data

- SEC EDGAR
- XBRL Company Facts
- financial metric normalization
- period-aware comparisons
- financial document processing
- deterministic financial calculations

### Software Engineering

- Python
- FastAPI
- REST API design
- Pydantic validation
- modular architecture
- error handling
- pytest
- GitHub Actions CI

### Data Engineering

- API-based data ingestion
- document parsing
- structured + unstructured data processing
- metadata preservation
- caching
- local index persistence

### MLOps / Deployment

- Docker
- Docker Compose
- configuration management
- structured logging
- latency/token/cost telemetry
- evaluation pipelines
- CI automation

---

## License

This project is distributed under the repository's MIT License.

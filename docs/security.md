# Security

## Secrets

Secrets are read from environment variables. `.env` is ignored by Git and `.env.example` contains placeholders only. API keys are never logged.

## Input validation

Pydantic validates ticker/form/query inputs and FastAPI middleware rejects oversized request bodies. Query length and tool-call budgets are separately bounded.

## SEC URL / SSRF protection

`SECClient` only accepts HTTPS URLs whose host is exactly `www.sec.gov` or `data.sec.gov`. User-controlled arbitrary URLs are never fetched. Embedded credentials and non-standard ports are rejected.

## Safe HTML parsing

BeautifulSoup/lxml parse downloaded filing HTML. Scripts, styles, SVG and noscript elements are discarded. Filing HTML is never executed.

## Prompt injection

Filing content is data, not instructions. The parser flags instruction-like phrases and the RAG system prompt explicitly requires the model to ignore any instructions appearing inside filing content.

## Rate limits

The SEC client throttles requests below the SEC fair-access maximum and uses exponential-backoff retries for transient errors and HTTP 429 responses.

## Tool-call limits

The query router is bounded and deterministic. It does not permit uncontrolled recursive tool use; `MAX_TOOL_CALLS` defaults to five.

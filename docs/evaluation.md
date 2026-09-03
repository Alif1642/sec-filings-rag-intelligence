# Evaluation

The repository includes an offline-compatible evaluation harness plus dynamic SEC ground truth for financial questions.

## Retrieval metrics

- Recall@1, Recall@3, Recall@5, Recall@10
- MRR
- NDCG@10

## Generation metrics

The default offline judge emits RAGAS-compatible metric names:

- answer faithfulness
- context precision
- context recall
- answer relevance

The implementation is intentionally replaceable with a model-based or RAGAS-backed judge without changing the runner contract.

## Citation metrics

- citation correctness: referenced IDs must exist in the supplied source set
- citation completeness: factual-looking answer sentences should be cited
- citation source validity: source URLs must resolve to approved SEC hosts

## Financial extraction metrics

For cases containing `expected_metric`, the evaluator fetches current ground truth directly from SEC Company Facts and reports:

- exact match
- absolute error
- percentage error

No static financial answer is fabricated or committed to the repository.

## System metrics

The evaluation runner reports average, P50 and P95 latency. Query-level cost can be estimated when provider pricing is configured.

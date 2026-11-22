# Retrieval benchmark MVP

From the project directory, after installing the package dependencies:

```bash
PYTHONPATH=src python benchmarks/run.py --top-k 1 --threshold 0.65
```

The JSON report compares cosine-only retrieval with dendritic retrieval on a
small synthetic dataset. Both use the same deterministic offline hash
embeddings and the same active-record store; superseded and retracted records
are excluded by both methods. This is a retrieval-only baseline, not a
complete RAG pipeline or an answer-generation benchmark.

The report includes micro-averaged recall@k, the fraction of labeled
undesirable records returned, and source-reference accuracy among retrieved
relevant records. A metric is `null` when its denominator is zero. Inspect
per-case results as well: the three toy cases are not evidence of real-world
performance or statistical significance.

Use `--cases path/to/cases.json` for additional data. Each case contains
`name`, `query`, `project`, a `memories` list with unique case-local IDs and
fields accepted by `ingest` (plus optional `verification_status` and
`conflict_status`), `relevant` mapping IDs to expected source references, and
an `avoid` list of undesirable IDs. The offline hash embedding is intended for
reproducibility, not production quality.

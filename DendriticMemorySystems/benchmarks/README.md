# Benchmarks

This directory contains structured benchmark cases for evaluating all three retrieval systems:

| System | Description |
|---|---|
| Baseline RAG | Cosine similarity only |
| Metadata-filtered RAG | Cosine + date/source/project filters |
| Dendritic Memory System | Full 7-branch scoring + graph + conflict |

## Categories

| Folder | What it tests |
|---|---|
| `episodic_memory/` | Retrieval of specific past events, conversations, decisions |
| `project_memory/` | Retrieval of design goals, milestones, and architecture choices |
| `temporal_updates/` | Current vs. superseded facts — staleness detection |
| `contradiction_cases/` | Suppression or labeling of known conflicting records |
| `provenance_cases/` | Prioritization of verified over unverified sources |

## Benchmark Format

Each subfolder should contain:

- `cases.json` — list of `{query, relevant_ids, notes}` entries
- `memories.json` — the memory records used in the benchmark
- `README.md` — description of the benchmark category

## Running Benchmarks

```bash
# Coming soon — evaluation harness in src/dendritic_memory/evaluation/
pytest benchmarks/ -v
```

See [`docs/evaluation_plan.md`](../docs/evaluation_plan.md) for the full evaluation design.

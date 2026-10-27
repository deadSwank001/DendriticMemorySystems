# Contributing to DendriticMemorySystems

Thank you for your interest in contributing. This project is a research prototype — contributions that improve empirical rigor, retrieval quality, and benchmark coverage are especially welcome.

## Getting Started

```bash
git clone https://github.com/deadSwank001/DendriticMemorySystems.git
cd DendriticMemorySystems
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
pytest
```

## Code Style

- Formatter: `black` (line length 100)
- Linter: `ruff`
- Type checker: `mypy`

Run all checks:

```bash
ruff check src/ tests/
black src/ tests/
mypy src/
```

## Contribution Guidelines

### What to contribute

- New or improved branch scorers in `src/dendritic_memory/scoring/`
- Benchmark cases in `benchmarks/` (episodic, temporal, contradiction, provenance)
- Evaluation metrics in `src/dendritic_memory/evaluation/`
- Vector store backends (Qdrant, Chroma, Pinecone) in `src/dendritic_memory/storage/`
- Consolidation improvements in `src/dendritic_memory/consolidation/`
- LLM adapter integrations in `src/dendritic_memory/interfaces/`

### Scope limits

This is a **software and evaluation project**. Contributions should not:

- Claim that the system is biologically equivalent to a neural system
- Introduce analogies that conflate software components with biological structures beyond the bounded design inspiration described in [`docs/biological_scope_limits.md`](docs/biological_scope_limits.md)
- Make performance claims that are not backed by benchmark results

### Pull Request Checklist

- [ ] Tests added or updated
- [ ] `ruff`, `black`, and `mypy` pass
- [ ] Docstrings updated for any changed public API
- [ ] Benchmark results included for any performance claim

## Reporting Issues

Please open a GitHub issue with a minimal reproducible example.

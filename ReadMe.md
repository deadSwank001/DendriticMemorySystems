# DendriticMemorySystems

**Bio-Inspired Persistent Storage, Associative Retrieval, and Synthetic-Cell Design Principles for Large Language Models**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Research Prototype](https://img.shields.io/badge/status-research--prototype-orange)](docs/concept_note.md)

> **Author:** David Burmeister — [github.com/deadSwank001](https://github.com/deadSwank001) · davidburm@gmail.com  
> **Predecessor project:** [deadSwank001/MemorySystemsStudy](https://github.com/deadSwank001/MemorySystemsStudy)

---

## Overview

Large language models reason effectively over supplied context but do not inherently provide **durable, transparent, source-grounded, user-governed memory** across sessions. Standard retrieval-augmented generation (RAG) relies on flat vector similarity search, which collapses temporal relevance, source reliability, contradiction status, task fit, and provenance into a single nearest-neighbor score.

**DendriticMemorySystems** introduces an external persistent-memory architecture inspired by a bounded computational analogy to dendritic integration: evaluate distinct forms of evidence locally before allowing a memory trace to influence global language-model behavior.

The system is **external to** and does not modify any LLM's internal weights, attention heads, or hidden states.

---

## Core Hypothesis

> A persistent LLM memory architecture based on **distributed local evaluation, associative activation, selective gating, provenance-aware plasticity, and episodic-to-semantic consolidation** will provide more context-sensitive and reliable memory than flat similarity retrieval alone.

---

## Architecture at a Glance

```
User interaction
      ↓
Task & context analysis
      ↓
Query embedding + metadata extraction
      ↓
Candidate memory retrieval
      ↓
┌─────────────────────────────────────────┐
│        Dendritic Branch Evaluation      │
│                                         │
│  Semantic   · Does this match meaning?  │
│  Temporal   · Is this current?          │
│  Contextual · Does it fit this task?    │
│  Relational · Is it graph-linked?       │
│  Provenance · Is it source-supported?  │
│  Conflict   · Is it contradicted?       │
│  Importance · Was it marked salient?    │
└─────────────────────────────────────────┘
      ↓
Activation threshold test
      ↓
 Retrieve ──or── Suppress
into LLM context    silently
```

See [`docs/architecture.md`](docs/architecture.md) for the full design.

---

## Memory Record

Each memory trace carries explicit, inspectable metadata:

```json
{
  "memory_id": "mem_000123",
  "content": "David Burmeister is developing Dendritic Memory for LLMs.",
  "memory_type": "project_fact",
  "created_at": "2026-09-28T17:33:00-05:00",
  "source_type": "user_provided",
  "confidence": 0.95,
  "verification_status": "user_confirmed",
  "conflict_status": "none",
  "retention_policy": "persistent"
}
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/deadSwank001/DendriticMemorySystems.git
cd DendriticMemorySystems
pip install -e ".[dev]"
```

### Minimal example

```python
from dendritic_memory import DendriticMemorySystem

dms = DendriticMemorySystem()

# Store a memory
dms.ingest(
    content="The project uses multi-branch scoring for retrieval.",
    source_type="user_provided",
    project_tags=["DendriticMemorySystems"],
)

# Retrieve with multi-branch activation
results = dms.retrieve(
    query="How does retrieval work in this project?",
    task_context="proposal_writing",
)

for r in results:
    print(r.content, "—", r.confidence)
```

See [`examples/`](examples/) for richer usage patterns.

---

## Repository Structure

```
DendriticMemorySystems/
├── README.md
├── LICENSE
├── pyproject.toml
├── docs/
│   ├── concept_note.md
│   ├── architecture.md
│   ├── evaluation_plan.md
│   ├── biological_scope_limits.md
│   └── what_this_proposal_does_not_claim.md
├── src/
│   └── dendritic_memory/
│       ├── ingestion/        # Event encoding and memory normalization
│       ├── storage/          # MemoryRecord schema and persistent store
│       ├── embeddings/       # Embedding models and vector indexing
│       ├── retrieval/        # Candidate retrieval and context packaging
│       ├── scoring/          # Branch scorers and integration rules
│       ├── graph/            # Associative memory graph
│       ├── provenance/       # Trust tracking and citation lineage
│       ├── consolidation/    # Episodic-to-semantic consolidation
│       ├── evaluation/       # Metrics and ablation harness
│       └── interfaces/       # Public API and LLM adapters
├── tests/
├── benchmarks/
├── notebooks/
└── examples/
```

---

## Comparison: Baseline RAG vs. Dendritic Memory

| Criterion | Baseline RAG | Dendritic Memory |
|---|---|---|
| Retrieval signal | Cosine similarity | 7-branch weighted scoring |
| Temporal awareness | None | Explicit recency + validity |
| Provenance tracking | None | Mandatory per-record |
| Contradiction handling | None | Conflict branch + penalty |
| Graph associations | None | Bounded spreading activation |
| Inspectability | Low | Full per-field audit trail |
| User correction | Overwrite | Versioned revision history |

---

## Evaluation Plan

Three systems are compared:

1. **Baseline RAG** — standard embedding similarity ranking
2. **Metadata-filtered RAG** — adds date, source type, and project filters
3. **Dendritic Memory System** — full multi-branch scoring + graph + provenance

See [`docs/evaluation_plan.md`](docs/evaluation_plan.md) for metrics and ablation design.

---

## Biological Inspiration — Scope Limits

Biological and synthetic-cell analogies are treated as **bounded design inspiration only**. The project does **not** claim:

- That software branches are biological dendrites
- That memory records are biological neurons
- That any synthetic-cell system (e.g., SpudCell) performs neuronal computation
- That the proposed system is biologically equivalent to neural tissue

See [`docs/biological_scope_limits.md`](docs/biological_scope_limits.md) and  
[`docs/what_this_proposal_does_not_claim.md`](docs/what_this_proposal_does_not_claim.md).

---

## Roadmap

- [x] Repository scaffold and documentation
- [ ] Minimal baseline RAG prototype
- [ ] `MemoryRecord` schema and in-memory store
- [ ] Sentence-transformer embeddings integration
- [ ] Multi-branch scoring (semantic + temporal + provenance)
- [ ] Associative graph with bounded spreading activation
- [ ] Conflict detection and revision policies
- [ ] Episodic-to-semantic consolidation
- [ ] Ablation benchmark suite
- [ ] Jupyter notebook walkthroughs

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you build on this work, please cite:

```bibtex
@software{burmeister2026dendritic,
  author    = {David Burmeister},
  title     = {DendriticMemorySystems: Bio-Inspired Persistent Storage and Associative Retrieval for LLMs},
  year      = {2026},
  url       = {https://github.com/deadSwank001/DendriticMemorySystems}
}
```

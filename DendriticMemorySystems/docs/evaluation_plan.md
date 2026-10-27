# Evaluation Plan

## Goal

Demonstrate — through **controlled empirical comparison and ablation studies** — whether multi-branch dendritic retrieval improves on flat similarity search for long-running LLM memory tasks.

Claims must be falsifiable. No improvements are assumed.

---

## Three Systems Under Comparison

| System | Description |
|---|---|
| **Baseline RAG** | Embedding retrieval using cosine similarity ranking only |
| **Metadata-filtered RAG** | Cosine retrieval + filters for date, source type, project, and verification status |
| **Dendritic Memory System** | Multi-branch scoring + associative graph activation + provenance gating + conflict handling |

---

## Benchmark Task Categories

| Category | Description | Location |
|---|---|---|
| Episodic memory | Retrieve specific past events, conversations, or decisions | `benchmarks/episodic_memory/` |
| Project memory | Retrieve design choices, milestones, goals, constraints | `benchmarks/project_memory/` |
| Temporal updates | Retrieve current vs. superseded facts | `benchmarks/temporal_updates/` |
| Contradiction cases | Suppress or label known conflicting records | `benchmarks/contradiction_cases/` |
| Provenance cases | Prioritize verified over unverified sources | `benchmarks/provenance_cases/` |

---

## Evaluation Questions

1. Does multi-branch scoring improve retrieval relevance vs. cosine similarity alone?
2. Does provenance-aware gating reduce unsupported or low-trust context in the LLM?
3. Does temporal scoring improve retrieval of current vs. superseded decisions?
4. Does a conflict branch reduce retrieval of known outdated or contradicted memories?
5. Does bounded graph activation improve recall for indirectly related but relevant information?
6. Does the system preserve user preferences without repeatedly surfacing temporary instructions?
7. Does the additional structure introduce unacceptable latency, complexity, or false suppression?

---

## Metrics

### Retrieval Quality
- **Precision@k** — fraction of top-k retrieved memories that are relevant
- **Recall@k** — fraction of relevant memories retrieved in top-k
- **MRR** — Mean Reciprocal Rank
- **nDCG** — Normalized Discounted Cumulative Gain
- **Human-rated contextual relevance** (1–5 Likert scale)

### Provenance and Trust
- **Citation completeness** — % of retrieved records with complete source metadata
- **Unverified recall rate** — % of retrieved records marked unverified or disputed
- **Contradicted memory rate** — % of retrieved records known to be contradicted

### Temporal Accuracy
- **Staleness rate** — % of retrieved records whose `valid_until` has passed
- **Recency gain** — improvement in recency rank vs. flat cosine retrieval

### System Performance
- **Retrieval latency** (ms) — end-to-end time from query to context package
- **Token cost** — tokens consumed by the constructed context package
- **False suppression rate** — % of relevant memories suppressed by branch gating
- **User correction rate** — % of returned memories requiring user correction
- **Memory-update accuracy** — correctness of conflict detection and revision writes

---

## Ablation Studies

Each ablation removes one component and measures performance impact. This identifies whether improvements come from the dendritic architecture, metadata filtering, graph structure, data hygiene, or a combination.

| Ablation | Component removed |
|---|---|
| A1 | Temporal branch |
| A2 | Provenance branch |
| A3 | Conflict penalty |
| A4 | Associative graph (relational branch) |
| A5 | Importance weighting |
| A6 | Replace multi-branch with pure cosine similarity |
| A7 | Replace tuned weights with equal weights (0.143 each) |
| A8 | Disable task-specific activation thresholds |
| A9 | Disable episodic-to-semantic consolidation |

See [`tests/test_scoring_ablation.py`](../tests/test_scoring_ablation.py) for the ablation harness.

---

## Reporting Standards

- Report mean ± standard deviation across benchmark folds
- Report per-category results (episodic, temporal, provenance, etc.) separately
- Report ablation deltas relative to full Dendritic Memory System
- Report latency at p50, p90, p99
- All benchmark data, evaluation scripts, and result logs will be committed to the repository

---

## Avoiding Overclaiming

The evaluation should measure **specific, bounded outcomes** rather than vague qualitative claims.

❌ "The system is more intelligent."  
❌ "The system thinks like a human."  
❌ "The system is biologically equivalent to a neural system."

✅ "Multi-branch scoring improves nDCG@10 by X% over cosine-only retrieval on temporal-update benchmarks."  
✅ "Conflict branch reduces contradicted-memory retrieval rate from X% to Y%."  
✅ "Provenance gating reduces unverified recall from X% to Y% with a false-suppression rate of Z%."

# Architecture: Dendritic Memory System

## System Overview

The Dendritic Memory System is an **external persistent-memory layer** positioned between a user, an LLM application, and a structured memory store.

```
User interaction / external event
         ↓
  Task & context analysis
         ↓
  Query embedding + metadata extraction
         ↓
  Candidate memory retrieval (broad)
         ↓
  ┌─────────────────────────────────────────────┐
  │          Dendritic Branch Evaluation        │
  │                                             │
  │  Semantic branch    · meaning match         │
  │  Temporal branch    · recency / validity    │
  │  Contextual branch  · task & goal fit       │
  │  Relational branch  · graph activation      │
  │  Provenance branch  · source trust          │
  │  Conflict branch    · contradiction penalty │
  │  Importance branch  · salience / priority   │
  └─────────────────────────────────────────────┘
         ↓
  Soma: weighted integration
         ↓
  Activation threshold test (default ≥ 0.65)
         ↓
  ┌──────────┴──────────┐
  ▼                     ▼
Retrieve            Suppress or
into LLM context    retain silently
```

---

## Five Principal Layers

### 1. Memory Ingestion and Normalization

Records user interactions, documents, observations, decisions, retrieved sources, and model-generated summaries while preserving source and authorship metadata.

**Key responsibilities:**
- Parse and normalize raw text, conversation turns, and documents
- Assign source type, author, and timestamp
- Flag model-generated vs. user-provided vs. externally cited content

### 2. Structured Memory Storage

Stores embeddings, metadata, timestamps, provenance, confidence, task context, graph relationships, contradiction links, and retention policies.

**Memory Record Schema:**

```json
{
  "memory_id": "mem_000123",
  "content": "...",
  "memory_type": "project_fact | episodic | semantic | procedural | preference",
  "embedding": "[float vector]",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "valid_from": "YYYY-MM-DD",
  "valid_until": null,
  "source_type": "user_provided | llm_generated | external_document | retrieved_source",
  "source_reference": "...",
  "author": "...",
  "confidence": 0.95,
  "verification_status": "user_confirmed | unverified | disputed | retracted",
  "importance": 0.85,
  "project_tags": [],
  "entities": [],
  "relations": [],
  "conflict_status": "none | disputed | superseded | retracted",
  "supersedes": null,
  "retention_policy": "persistent | session | expiring",
  "retrieval_count": 0,
  "last_retrieved": null,
  "version": 1
}
```

### 3. Associative Graph Layer

Represents relationships among projects, people, concepts, documents, decisions, tasks, claims, and episodes.

**Example graph topology:**

```
David Burmeister
│
├── develops ──► Dendritic Memory for LLMs
│
├── maintains ──► MemorySystemsStudy
│
└── may create ──► DendriticMemorySystems
                   │
                   ├── compares_against ──► Baseline RAG
                   ├── uses ──► Provenance-aware retrieval
                   ├── uses ──► Multi-branch scoring
                   └── investigates ──► Synthetic-cell design principles
                                        │
                                        └── bounded_context ──► SpudCell
```

**Activation rules:**
- Spreading activation is **bounded**: max depth 2–3 hops
- Score decays by relation type and hop distance
- Token budget enforced before LLM context construction

### 4. Dendritic Activation Layer

Evaluates candidate memories across specialized branches.

| Branch | Input | Decision question |
|---|---|---|
| Semantic | Embedding similarity, keywords, entities | "Is this about the queried topic?" |
| Temporal | Date, recency, validity interval | "Is this still current?" |
| Contextual | Active task, user goal, document section | "Does this help with the current task?" |
| Relational | Graph links and associated memories | "Is it connected to active concepts?" |
| Provenance | Source type, author, verification status | "Can this claim be trusted?" |
| Conflict | Contradicting records, correction flags | "Has this been revised or retracted?" |
| Importance | User preferences, retrieval count, priority | "Should this surface even if weakly similar?" |

**Default activation formula:**

```python
activation = (
    0.40 * semantic_score
  + 0.20 * temporal_score
  + 0.20 * context_score
  + 0.20 * provenance_score
  - 0.40 * conflict_penalty
)
retrieve = activation >= 0.65
```

Weights and threshold are configurable and may be learned from retrieval feedback.

**Branch-specific nonlinear behavior examples:**

```
High semantic similarity
  + weak provenance
  + known contradiction
  = suppress or label as disputed

Moderate semantic similarity
  + explicit user instruction
  + active project context
  + recent confirmed decision
  = retrieve with high priority
```

### 5. LLM Context-Construction Layer

Selects, annotates, orders, and compresses activated memories before placing them into the LLM context window.

**Responsibilities:**
- Rank activated memories by composite score
- Annotate uncertain or disputed memories with explicit labels
- Respect token budget (configurable)
- Include provenance citations in the context package
- Flag any records approaching expiry or marked for review

---

## Memory Types and Consolidation

| Type | Purpose | Example |
|---|---|---|
| Episodic | Time-bound record of a specific event | "On Sept 28, user chose dendritic framing." |
| Semantic | Generalized representation from multiple episodes | "User is developing bio-inspired LLM memory." |
| Procedural | Reusable workflow or implementation instruction | "Run ablations before claiming improvements." |
| Preference | User-confirmed enduring preference | "Credit work to David Burmeister." |
| Project | Goals, design choices, milestones, constraints | "Create DendriticMemorySystems as a separate repo." |

### Consolidation Lifecycle

```
Experience
    ↓
Candidate episodic trace
    ↓
Gated storage decision
    ↓
Association with related traces
    ↓
Repeated retrieval, confirmation, or relevance
    ↓
Consolidation trigger
    ↓
Semantic abstraction — linked to source episodes
```

> **Invariant:** Consolidated semantic memories must remain traceable to their source episodic records. Provenance is never erased by consolidation.

---

## Core Policy Choices

| Policy | Description |
|---|---|
| Selective persistence | Filter routine dialogue; do not commit every interaction to durable storage |
| Provenance tracking | Source, actor, date, document origin, derivation history are mandatory |
| Epistemic separation | Direct evidence, user assertion, external source, LLM inference, and system synthesis are distinct categories |
| Contradiction preservation | Contradictions are stored and linked, not silently overwritten |
| Deletion as first-class op | Records can be corrected, superseded, expired, or deleted with a revision trail |
| Bounded graph activation | Depth limits, score decay, relation-type filters, and token budgets prevent runaway retrieval |

# Concept Note: Dendritic Memory for Large Language Models

**Author:** David Burmeister  
**Project:** DendriticMemorySystems  
**Status:** Research Prototype — v0.1

---

## Problem

Large language models are effective at language generation and contextual reasoning, but they do not inherently provide **durable, transparent, source-grounded, user-governed memory** across sessions.

Conventional retrieval-augmented generation (RAG) relies primarily on flat vector similarity search. A single similarity score collapses:

- Temporal relevance vs. semantic proximity
- Verified evidence vs. unsupported inference
- Current decisions vs. superseded ones
- User preferences vs. temporary task instructions
- Confirmed facts vs. known contradictions

This conflation degrades retrieval quality for long-running research, writing, planning, and personalized-assistant tasks.

---

## Central Hypothesis

> A persistent LLM memory architecture based on **distributed local evaluation, associative activation, selective gating, provenance-aware plasticity, and episodic-to-semantic consolidation** will provide more context-sensitive and reliable memory than flat similarity retrieval alone.

---

## Key Insight

Biological neurons — to a bounded degree — evaluate distributed dendritic inputs locally before their broader output is determined. The relevant computational lesson is narrow but useful:

> **Local computation can occur before global activation.**

This project extracts that principle into explicit software: before a memory trace is retrieved into an LLM context, it must satisfy an inspectable multi-branch decision process rather than simply pass a cosine similarity threshold.

---

## What the System Does

1. **Ingests** interactions, documents, decisions, and external sources as structured episodic records
2. **Stores** those records with mandatory provenance, timestamps, confidence, and conflict metadata
3. **Evaluates** candidate memories across seven branches: semantic, temporal, contextual, relational, provenance, conflict, and importance
4. **Activates** associated records through a bounded associative graph
5. **Constructs** a compact, annotated evidence package for the LLM context window

---

## Future-Work Hypothesis (Speculative)

> Principles from synthetic-cell organization may supply useful abstractions for computational memory, but their relevance must be demonstrated through clearly bounded modeling and experimental work rather than assumed by analogy.

This hypothesis does **not** assert that any synthetic-cell system performs neuronal computation, implements LLM memory, or is a demonstrated cellular memory computer.

---

## Immediate Deliverable

A reproducible Python prototype comparing:

| System | Description |
|---|---|
| Baseline RAG | Standard cosine-similarity retrieval |
| Metadata-filtered RAG | Adds date, source type, and project filters |
| Dendritic Memory System | Full multi-branch scoring + associative graph + conflict handling |

---

## Relationship to Prior Work

This project extends [MemorySystemsStudy](https://github.com/deadSwank001/MemorySystemsStudy), which frames AI memory as a problem of persistent storage, semantic representation, embeddings, and retrieval beyond an LLM's immediate context window.

The present work asks whether the storage and retrieval substrate itself should be organized as a **distributed, locally selective, plastic associative system** rather than a flat vector database alone.

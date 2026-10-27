# Biological Scope Limits

## Purpose

This document defines the **explicit limits of biological analogy** used in this project. It is intended for collaborators, reviewers, and readers who may wish to understand what the project does and does not claim about biological systems.

---

## The Analogy Used

The term **"dendritic"** in this project refers to a bounded architectural analogy only.

In biological neurons, dendritic branches receive signals across distributed compartments. Local dendritic regions can perform nonlinear integration before a cell's broader output is determined.

The **computational lesson** extracted is intentionally narrow:

> **Local computation can occur before global activation.**

This does not require that software branches are biological dendrites, or that memory records are biological neurons.

---

## What the Analogy Contributes

| Biological term | Software abstraction used here |
|---|---|
| Dendritic branch | Specialized local evaluator of one evidence dimension |
| Synapse | Weighted association between memory traces |
| Soma | Aggregation and memory-activation decision point |
| Action potential | Memory becomes eligible for retrieval |
| Plasticity | Updating confidence, relevance, or link strength |
| Neural population | Cluster or graph neighborhood of related memories |
| Consolidation | Compression of repeated episodes into semantic memory |

These abstractions are used **only** to organize software architecture. They are not claims that the software reproduces biological computation.

---

## Explicitly Bounded Claims

### Dendrites

- Software retrieval branches are **not** biological dendrites
- The project does not model dendritic ion channels, cable theory, compartmental biophysics, or synaptic plasticity in any biological sense
- "Dendritic-inspired" refers only to the principle of local evaluation before global activation

### Artificial Neurons vs. Biological Neurons

- An artificial neuron (hidden unit in an LLM feed-forward network) is **not** a biological neuron
- The project does not claim that internal LLM units are equivalent to neurons, dendrites, or any biological structure
- No attempt is made to locate, decode, map, or reverse-engineer internal LLM parameters

### LLM Memory vs. Neural Memory

- The proposed external memory store is **not** a model of biological memory
- The project does not claim to reproduce hippocampal consolidation, synaptic long-term potentiation, or any biological memory mechanism
- "Episodic-to-semantic consolidation" refers to a software policy for compressing specific records into general ones — it is named by analogy, not by identity

### Synthetic Cell Context (SpudCell)

The project references SpudCell as a bounded scientific context for a speculative future-work hypothesis only. The project does **not** claim:

- That SpudCell is a neural system
- That SpudCell performs neuronal computation
- That SpudCell is an LLM-memory platform
- That synthetic cells have been demonstrated to function as general-purpose cellular memory computers
- That any wet-lab or synthetic-cell work is a prerequisite for the software prototype
- That a computational collaboration with any synthetic-biology group is appropriate without mutual scientific interest and clearly defined scope

---

## Why Biological Inspiration Is Still Useful

Biological systems provide a source of **computational design questions**, not engineering proofs:

- Can local evaluation improve global selection?
- Can memory retrieval use multiple evidence types rather than one similarity score?
- Can persistent state be structured, revisable, and resource-aware?
- Can selective thresholds reduce irrelevant or unsupported recall?
- Can associative relationships support context-sensitive retrieval?
- Can confidence, contradiction, and provenance be represented explicitly?

Whether any of these design choices produce measurable improvement must be shown **empirically through benchmarking and ablation**, not inferred from the biological analogy alone.

---

## Summary Statement

> Biological inspiration is useful only when its limits are explicit. The purpose of this project is not to copy biology literally, claim equivalence between software and neural tissue, or imply that a synthetic-cell system performs the same functions as a neural network.
>
> The proposed system is fundamentally a **software and evaluation project**. Its value depends on whether it can demonstrate measurable improvements in retrieval quality, provenance awareness, temporal accuracy, contradiction handling, user control, and inspectability compared with conventional RAG.

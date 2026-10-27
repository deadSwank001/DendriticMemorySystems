"""
Proposal-writing example.

Demonstrates how the Dendritic Memory System retrieves context-sensitive
memories during a long-running writing task, correctly prioritizing verified
project decisions over tangentially related content.
"""

from dendritic_memory import DendriticMemorySystem

# ---- Setup ---------------------------------------------------------------

dms = DendriticMemorySystem(activation_threshold=0.3)

# Ingest project knowledge base
dms.ingest(
    "The architecture uses seven branches: semantic, temporal, contextual, "
    "relational, provenance, conflict, and importance.",
    memory_type="project",
    confidence=0.95,
    importance=0.9,
    project_tags=["DendriticMemorySystems"],
    entities=["DendriticMemorySystems"],
)

dms.ingest(
    "Biological inspiration is treated as a bounded analogy only. "
    "Software branches are not biological dendrites.",
    memory_type="project",
    confidence=0.98,
    importance=0.95,
    project_tags=["DendriticMemorySystems"],
)

dms.ingest(
    "SpudCell is a synthetic-cell research project at the University of Minnesota. "
    "It is not claimed to be a neural system or LLM-memory platform.",
    memory_type="project",
    confidence=0.97,
    importance=0.9,
    project_tags=["DendriticMemorySystems", "SpudCell"],
    entities=["SpudCell"],
)

stale_id = dms.ingest(
    "The project was originally called MemoryNet.",
    memory_type="episodic",
    confidence=0.6,
    importance=0.3,
    project_tags=["DendriticMemorySystems"],
).memory_id

# Supersede the stale record
dms.supersede(
    stale_id,
    "The project is named DendriticMemorySystems, not MemoryNet.",
    memory_type="project",
    confidence=0.99,
    project_tags=["DendriticMemorySystems"],
)

# ---- Retrieval -----------------------------------------------------------

print("\n=== Query: 'What does the architecture consist of?' ===")
results = dms.retrieve(
    query="What does the architecture consist of?",
    task_context="proposal_writing",
    active_project="DendriticMemorySystems",
)

for r in results:
    print(f"  [{r.activation:.3f}] {r.record.content[:90]}")
    print(f"          sem={r.scores.semantic:.2f} | "
          f"prov={r.scores.provenance:.2f} | "
          f"conflict_pen={r.scores.conflict_penalty:.2f}")

print("\n=== Query: 'Is SpudCell a neural system?' ===")
results = dms.retrieve(
    query="Is SpudCell a neural system?",
    task_context="proposal_writing",
    active_project="DendriticMemorySystems",
    active_entities=["SpudCell"],
)
for r in results:
    print(f"  [{r.activation:.3f}] {r.record.content[:90]}")

print("\n=== System stats ===")
print(dms.stats())

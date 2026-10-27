"""
Project assistant example.

Demonstrates how the system preserves project decisions across multiple
sessions and correctly suppresses stale or superseded records.
"""

from dendritic_memory import DendriticMemorySystem

dms = DendriticMemorySystem(activation_threshold=0.2)

# Session 1: initial design decisions
dms.ingest(
    "Evaluation will compare baseline RAG, metadata-filtered RAG, and dendritic scoring.",
    memory_type="project",
    confidence=0.95,
    importance=0.9,
    project_tags=["DendriticMemorySystems"],
)

dms.ingest(
    "The default activation threshold is 0.65.",
    memory_type="project",
    confidence=0.9,
    importance=0.7,
    project_tags=["DendriticMemorySystems"],
)

old_decision = dms.ingest(
    "The project will use Chroma as the vector store.",
    memory_type="project",
    confidence=0.75,
    importance=0.6,
    project_tags=["DendriticMemorySystems"],
)

# Session 2: revised decision
dms.supersede(
    old_decision.memory_id,
    "The project will use FAISS for initial prototyping; Qdrant for production.",
    memory_type="project",
    confidence=0.92,
    importance=0.8,
    project_tags=["DendriticMemorySystems"],
)

# Session 3: query
print("=== Query: 'What vector store are we using?' ===")
results = dms.retrieve(
    "What vector store are we using?",
    active_project="DendriticMemorySystems",
)
for r in results:
    status = r.record.conflict_status.value
    print(f"  [{r.activation:.3f}] [{status}] {r.record.content[:90]}")

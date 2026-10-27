"""
User preference example.

Demonstrates how user-confirmed enduring preferences are stored with
high importance and retrieved with priority over temporary instructions.
"""

from dendritic_memory import DendriticMemorySystem

dms = DendriticMemorySystem(activation_threshold=0.2)

# Durable preferences
dms.ingest(
    "Credit all academic and research work to David Burmeister.",
    memory_type="preference",
    source_type="user_provided",
    confidence=0.99,
    importance=1.0,
    project_tags=["DendriticMemorySystems"],
    entities=["David Burmeister"],
)

dms.ingest(
    "Always include the scope-limits appendix in any collaborator-facing document.",
    memory_type="preference",
    source_type="user_provided",
    confidence=0.98,
    importance=0.95,
    project_tags=["DendriticMemorySystems"],
)

# Temporary instruction (low importance, session retention)
dms.ingest(
    "For this session: format output as plain text, not markdown.",
    memory_type="episodic",
    source_type="user_provided",
    confidence=0.9,
    importance=0.3,
    retention_policy="session",
    project_tags=["DendriticMemorySystems"],
)

# Query preferences
print("=== Query: 'What are the user's persistent preferences?' ===")
results = dms.retrieve(
    "What are the user's persistent preferences for formatting and attribution?",
    active_project="DendriticMemorySystems",
)
for r in results:
    print(
        f"  [{r.activation:.3f}] [{r.record.memory_type.value}] "
        f"[imp={r.record.importance:.2f}] {r.record.content[:90]}"
    )

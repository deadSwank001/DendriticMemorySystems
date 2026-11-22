"""Compare retrieval on a small, reproducible synthetic benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dendritic_memory import DendriticMemorySystem
from dendritic_memory.embeddings.embedder import EmbeddingLayer
from dendritic_memory.storage.memory_record import ConflictStatus, VerificationStatus

CASES = Path(__file__).with_name("cases.json")


def evaluate(cases: list[dict], top_k: int, threshold: float) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    results = {}
    for method in ("cosine", "dendritic"):
        details = []
        found_total = relevant_total = avoided_total = avoid_returned = 0
        sourced_total = sourced_correct = 0
        for case in cases:
            if not case["relevant"]:
                raise ValueError(f"{case['name']}: expected at least one relevant memory")
            system = DendriticMemorySystem(activation_threshold=threshold)
            # Use the same offline embeddings for both methods and all runs.
            system.embedder.embed = EmbeddingLayer._fallback_embed
            records = {}
            for memory in case["memories"]:
                fields = dict(memory)
                case_id = fields.pop("id")
                verification = fields.pop("verification_status", "unverified")
                conflict = fields.pop("conflict_status", "none")
                record = system.ingest(**fields)
                record.verification_status = VerificationStatus(verification)
                record.conflict_status = ConflictStatus(conflict)
                records[record.memory_id] = case_id

            if method == "cosine":
                retrieved = system.retrieve_baseline(case["query"], top_k=top_k)
            else:
                retrieved = system.retrieve(case["query"], top_k=top_k, active_project=case["project"])
            ids = [records[item.record.memory_id] for item in retrieved]
            relevant = case["relevant"]
            relevant_hits = [item for item in retrieved if records[item.record.memory_id] in relevant]
            found = len(relevant_hits)
            correct_sources = sum(
                item.record.source_reference == relevant[records[item.record.memory_id]]
                for item in relevant_hits
            )
            leaks = len(set(ids) & set(case["avoid"]))
            found_total += found
            relevant_total += len(relevant)
            avoid_returned += leaks
            avoided_total += len(case["avoid"])
            sourced_total += found
            sourced_correct += correct_sources
            details.append({"case": case["name"], "retrieved": ids, "relevant_found": found,
                            "avoid_returned": leaks, "correct_sources": correct_sources})
        results[method] = {
            "recall_at_k": found_total / relevant_total if relevant_total else None,
            "avoid_leak_rate": avoid_returned / avoided_total if avoided_total else None,
            "source_accuracy_on_relevant_hits": sourced_correct / sourced_total if sourced_total else None,
            "cases": details,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    print(json.dumps({"top_k": args.top_k, "threshold": args.threshold,
                      "embedding": "offline hash fallback", "results": evaluate(cases, args.top_k, args.threshold)},
                     indent=2))


if __name__ == "__main__":
    main()

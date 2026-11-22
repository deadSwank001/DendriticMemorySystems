import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "benchmarks" / "run.py"
spec = importlib.util.spec_from_file_location("benchmark_runner", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_metrics_include_leaks_and_source_errors():
    cases = [{
        "name": "fixture",
        "query": "same text",
        "project": "p",
        "memories": [
            {"id": "bad", "content": "same text", "source_reference": "bad", "project_tags": ["p"]},
            {"id": "good", "content": "different", "source_reference": "actual", "project_tags": ["p"]},
        ],
        "relevant": {"good": "expected"},
        "avoid": ["bad"],
    }]
    result = benchmark.evaluate(cases, top_k=2, threshold=0.0)
    for metrics in result.values():
        assert metrics["recall_at_k"] == 1
        assert metrics["avoid_leak_rate"] == 1
        assert metrics["source_accuracy_on_relevant_hits"] == 0


def test_empty_results_and_invalid_k():
    cases = [{
        "name": "fixture", "query": "test", "project": "p",
        "memories": [{"id": "one", "content": "test"}],
        "relevant": {"one": "source"}, "avoid": [],
    }]
    result = benchmark.evaluate(cases, top_k=1, threshold=10)
    assert result["dendritic"]["recall_at_k"] == 0
    assert result["dendritic"]["source_accuracy_on_relevant_hits"] is None
    assert result["dendritic"]["avoid_leak_rate"] is None
    with pytest.raises(ValueError):
        benchmark.evaluate(cases, top_k=0, threshold=0)

"""
Tests for the seed aggregation in results/benchmark_matrix.py.
Pure Python (numpy, pandas, sqlite3): no mlflow, torch or GPU needed.

Usage (from the repository root):
    python -m pytest tests/test_benchmark_matrix.py
    python tests/test_benchmark_matrix.py
"""

import os
import sqlite3
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "results"))

import numpy as np
import pandas as pd

from benchmark_matrix import aggregate_benchmark_runs


def _run(seed, start_time, accuracy, latency_ms=100.0, model="phi-4-mini-instruct", task="classification"):
    return {"model": model, "task": task, "seed": seed, "start_time": start_time,
            "accuracy": accuracy, "latency_ms": latency_ms, "cost_per_1m_tokens": 1.0,
            "privacy_risk": "low"}


def test_seeds_are_averaged_with_sample_sd():
    runs = [
        _run(None, 1, 0.50),                 # thesis-era unseeded run: superseded
        _run(42, 2, 0.90, latency_ms=100.0),
        _run(43, 3, 0.10),                   # stale, re-run below
        _run(43, 4, 0.92, latency_ms=110.0),
        _run(44, 5, 0.94, latency_ms=120.0),
    ]
    [cell] = aggregate_benchmark_runs(runs)
    assert cell["n_seeds"] == 3 and cell["seeds"] == "42;43;44"
    assert np.isclose(cell["accuracy"], 0.92)
    assert np.isclose(cell["accuracy_sd"], 0.02)
    assert np.isclose(cell["latency_ms"], 110.0)
    assert np.isclose(cell["latency_ms_sd"], 10.0)


def test_unseeded_cell_keeps_its_latest_run():
    [cell] = aggregate_benchmark_runs([_run(None, 1, 0.70), _run(None, 2, 0.80)])
    assert cell["accuracy"] == 0.80
    assert cell["n_seeds"] == 1 and cell["seeds"] == "" and cell["accuracy_sd"] is None


def test_cells_are_kept_apart():
    cells = aggregate_benchmark_runs([_run(42, 1, 0.9), _run(42, 1, 0.4, task="ner")])
    assert {(c["task"], c["accuracy"]) for c in cells} == {("classification", 0.9), ("ner", 0.4)}


def _thesis_records(db_path):
    """The phase2_benchmark runs in mlflow.db, in the record shape the compiler builds."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    runs = con.execute(
        "SELECT r.run_uuid, r.start_time FROM runs r JOIN tags t ON t.run_uuid = r.run_uuid "
        "WHERE t.key = 'phase' AND t.value = 'phase2_benchmark'"
    ).fetchall()
    records = []
    for run_uuid, start_time in runs:
        tags = dict(con.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (run_uuid,)))
        metrics = dict(con.execute("SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (run_uuid,)))
        records.append({
            "model": tags["model"].split("/")[-1], "task": tags["task"],
            "seed": int(tags["seed"]) if "seed" in tags else None, "start_time": start_time,
            "accuracy": metrics.get("accuracy"), "latency_ms": metrics.get("latency_ms"),
            "cost_per_1m_tokens": metrics.get("cost_per_1m_tokens"),
            "privacy_risk": tags.get("privacy_risk", "unknown"),
        })
    con.close()
    return records


def _released_slm_cells():
    released = pd.read_csv(os.path.join(REPO, "results", "benchmark_matrix.csv"))
    return released[released["method"] != "API"].set_index(["model", "task"])


def test_thesis_runs_reproduce_the_released_matrix():
    """
    With no seeded runs yet, aggregation must give back the published SLM cells.

    NER is the exception: its re-measurement (commit 04f2a77) updated the CSV
    and results/predictions/ but not the committed mlflow.db, which still holds
    the withdrawn pre-fix NER runs. test_ner_cells_match_saved_predictions
    covers those three cells instead.
    """
    cells = aggregate_benchmark_runs(_thesis_records(os.path.join(REPO, "mlflow.db")))
    got = {(c["model"], c["task"]): c for c in cells}
    released = _released_slm_cells()
    assert set(got) == set(released.index)
    for key, row in released.drop(index="ner", level="task").iterrows():
        assert np.isclose(got[key]["accuracy"], row["accuracy"]), key
        assert np.isclose(got[key]["latency_ms"], row["latency_ms"]), key
        assert got[key]["n_seeds"] == 1


def test_ner_cells_match_saved_predictions():
    for (model, task), row in _released_slm_cells().xs("ner", level="task", drop_level=False).iterrows():
        preds = pd.read_csv(os.path.join(REPO, "results", "predictions", f"{model}__ner.csv"))
        assert np.isclose(preds["f1"].mean(), row["accuracy"], atol=5e-5), model


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")

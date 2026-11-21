"""
Tests for src/evaluation/ner_typed.py and results/ner_typed_f1.py.
Needs pandas, numpy and seqeval; no torch or GPU. The end-to-end test also
needs the CoNLL-2003 test split (via `datasets` or its local HF cache).

Usage (from the repository root):
    python -m pytest tests/test_ner_typed.py
    python tests/test_ner_typed.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "results"))

import pandas as pd
from seqeval.metrics import f1_score as seqeval_f1

from evaluation.ner_typed import (
    bio_tags, bio_to_entities, lenient_sentence_f1, micro_prf, parse_api_entities,
)

TOKENS = ["SOCCER", "-", "JAPAN", "GET", "LUCKY", "WIN", ",", "CHINA", "IN", "SURPRISE", "DEFEAT", "."]
GOLD = ["O", "O", "B-LOC", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O"]  # CoNLL's own labels


def test_bio_decoding_keeps_types():
    assert bio_to_entities(TOKENS, GOLD) == [("LOC", "JAPAN"), ("PER", "CHINA")]
    assert bio_to_entities(["New", "York", "Times"], ["B-ORG", "I-ORG", "I-ORG"]) == [("ORG", "New York Times")]
    assert bio_to_entities(["a", "b"], ["I-PER", "O"]) == []  # orphan I- is O, as in the thesis decoder


def test_bio_tags_parses_generated_ids():
    assert bio_tags("[5, 0, 6]") == ["B-LOC", "O", "I-LOC"]
    assert bio_tags("no ids here") is None


def test_typing_is_what_separates_the_instruments():
    gold = [[("LOC", "JAPAN"), ("PER", "CHINA")]]
    pred = [[("LOC", "Japan"), ("LOC", "China")]]  # gpt-4o's answer for this sentence, roughly
    assert lenient_sentence_f1(gold[0], pred[0]) == 1.0
    assert micro_prf(gold, pred, typed=False)["f1"] == 1.0
    assert micro_prf(gold, pred)["f1"] == 0.5
    assert micro_prf(gold, pred, case_sensitive=True)["f1"] == 0.0


def test_micro_counts_duplicates():
    gold = [[("LOC", "China"), ("LOC", "China")]]
    assert micro_prf(gold, [[("LOC", "China")]])["recall"] == 0.5


def test_typed_micro_equals_seqeval_on_token_aligned_spans():
    pred = ["O", "O", "B-LOC", "O", "O", "O", "O", "B-LOC", "O", "O", "O", "O"]
    ours = micro_prf([bio_to_entities(TOKENS, GOLD)], [bio_to_entities(TOKENS, pred)], case_sensitive=True)
    assert ours["f1"] == seqeval_f1([GOLD], [pred]) == 0.5


def test_api_parsing():
    assert parse_api_entities('```json\n{"LOC": ["Japan"], "PER": []}\n```') == [("LOC", "Japan")]
    assert parse_api_entities('{"PER": ["Nadim Ladki"], "ORG": null}') == [("PER", "Nadim Ladki")]
    assert parse_api_entities('{"PER": ["truncated') is None


def test_released_scores_are_reproduced():
    """Recomputing the thesis's lenient score from the saved outputs must match every stored row."""
    from ner_typed_f1 import api_rows, load_conll_test, slm_rows
    os.chdir(REPO)
    conll = load_conll_test()
    rows = pd.DataFrame(slm_rows(conll) + api_rows(conll))
    assert (rows["lenient_rows_reproduced"] == rows["n"]).all()

    matrix = pd.read_csv("results/benchmark_matrix.csv")
    full = rows[(rows["arm"] == "API") | (rows["n"] == 200)]
    for _, r in full.iterrows():
        stored = matrix[(matrix["model"] == r["model"]) & (matrix["task"] == "ner")]["accuracy"].iloc[0]
        assert abs(r["lenient_f1"] - stored) < 1e-3, r["model"]

    slm = rows[rows["arm"] == "fine-tuned"]
    assert (slm["seqeval_f1"] == slm["typed_micro_f1_cased"]).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")

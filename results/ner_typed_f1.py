"""
Typed NER scores for both arms, recomputed from the saved outputs.

Inputs need no GPU or API calls: the fine-tuned arm's per-instance tag
outputs (results/predictions/*__ner.csv), the API arm's responses
(logs/conll_baseline.csv) and the CoNLL-2003 test split for typed gold.

For every model this reports, on the same footing:
- lenient_f1: the thesis instrument (sentence-averaged set F1, lowercased,
  types dropped), recomputed here; lenient_rows_reproduced counts the rows
  whose recomputed score equals the stored one, which validates the parsing.
- typed_micro_*: micro P/R/F1 over (type, surface) pairs, case-insensitive.
- typed_micro_f1_cased: the same, case-sensitive.
- untyped_micro_f1: micro F1 with types dropped, to isolate what typing costs.
- seqeval_f1: span F1 from seqeval over BIO sequences (fine-tuned arm only;
  the API arm's JSON output is not token-aligned).

The fine-tuned arm was evaluated on the first 200 test sentences and the API
arm on the first 100, so fine-tuned models are reported on both subsets and
the n=100 rows are the paired comparison.

Run from the repository root:
    python results/ner_typed_f1.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from seqeval.metrics import f1_score as seqeval_f1

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evaluation.ner_typed import (  # noqa: E402
    NER_ID2TAG, bio_tags, bio_to_entities, lenient_sentence_f1, micro_prf, parse_api_entities,
)

SLM_MODELS = ["phi-4-mini-instruct", "Llama-3.2-3B-Instruct", "Mistral-7B-v0.3"]
API_PREFIX = {"openai": "gpt-4o", "anthropic": "claude-haiku-4-5", "gemini": "gemini-2.5-flash"}
PROMPT_CHAR_LIMIT = 400  # SLMDataset._format_one_sample truncates the input text to this
SAVED_PREDICTION_CHARS = 300  # thesis-era evaluate_ner stored raw_prediction[:300]; it now keeps it whole


def load_conll_test() -> pd.DataFrame:
    """CoNLL-2003 test split (tokens, ner_tags), from `datasets` or its local cache."""
    try:
        from datasets import load_dataset
        ds = load_dataset("eriktks/conll2003", split="test", revision="refs/convert/parquet")
        return ds.to_pandas()[["tokens", "ner_tags"]]
    except ImportError:
        import pyarrow as pa
        pattern = os.path.expanduser(
            "~/.cache/huggingface/datasets/eriktks___conll2003/default/0.0.0/*/conll2003-test.arrow")
        path = sorted(glob.glob(pattern))[0]
        with open(path, "rb") as f:
            return pa.ipc.open_stream(f).read_all().to_pandas()[["tokens", "ner_tags"]]


def score(model: str, arm: str, gold, pred, lenient, lenient_stored, parse_failures, extra=None) -> dict:
    typed = micro_prf(gold, pred)
    row = {
        "model": model, "arm": arm, "n": len(gold),
        "lenient_f1": round(float(np.mean(lenient)), 4),
        "lenient_rows_reproduced": int(np.sum(np.isclose(lenient, lenient_stored, atol=5e-5))),
        "typed_micro_precision": round(typed["precision"], 4),
        "typed_micro_recall": round(typed["recall"], 4),
        "typed_micro_f1": round(typed["f1"], 4),
        "typed_micro_f1_cased": round(micro_prf(gold, pred, case_sensitive=True)["f1"], 4),
        "untyped_micro_f1": round(micro_prf(gold, pred, typed=False)["f1"], 4),
        "parse_failures": parse_failures,
    }
    row.update(extra or {})
    return row


def slm_rows(conll: pd.DataFrame, predictions_dir: str = "results/predictions") -> list:
    out = []
    for model in SLM_MODELS:
        preds = pd.read_csv(os.path.join(predictions_dir, f"{model}__ner.csv"))
        gold, pred, gold_tags, pred_tags, lenient, cut = [], [], [], [], [], []
        parse_failures = 0
        for _, r in preds.iterrows():
            tokens = list(conll.at[int(r["idx"]), "tokens"])
            tags = [NER_ID2TAG[t] for t in conll.at[int(r["idx"]), "ner_tags"]]
            seen = " ".join(tokens)[:PROMPT_CHAR_LIMIT].split()
            raw = str(r["raw_prediction"]) if pd.notna(r["raw_prediction"]) else ""
            ptags = bio_tags(raw)
            if ptags is None:
                parse_failures += 1
                ptags = []
            # A tag list longer than the stored prefix may have been cut off in the file.
            cut.append(len(raw) >= SAVED_PREDICTION_CHARS and len(ptags) < len(seen))
            p = bio_to_entities(seen, ptags)
            gold.append(bio_to_entities(tokens, tags))
            pred.append(p)
            lenient.append(lenient_sentence_f1(bio_to_entities(seen, tags), p))
            aligned = (ptags + ["O"] * len(seen))[:len(seen)]
            gold_tags.append(tags)
            pred_tags.append(aligned + ["O"] * (len(tokens) - len(seen)))
        stored = preds["f1"].to_numpy()
        for n in (len(preds), 100):
            out.append(score(
                model, "fine-tuned", gold[:n], pred[:n], lenient[:n], stored[:n],
                parse_failures=int(sum(bio_tags(str(x)) is None for x in preds["raw_prediction"][:n])),
                extra={"seqeval_f1": round(seqeval_f1(gold_tags[:n], pred_tags[:n]), 4),
                       "rows_cut_in_saved_file": int(sum(cut[:n]))},
            ))
    return out


def api_rows(conll: pd.DataFrame, log_path: str = "logs/conll_baseline.csv") -> list:
    log = pd.read_csv(log_path)
    # The log keeps only a prefix of each sentence (the full sentence was sent to the API).
    for i, text in enumerate(log["text"]):
        if not " ".join(conll.at[i, "tokens"]).startswith(text):
            raise ValueError(f"{log_path} row {i} is not CoNLL test sentence {i}")
    gold = [bio_to_entities(list(conll.at[i, "tokens"]), [NER_ID2TAG[t] for t in conll.at[i, "ner_tags"]])
            for i in range(len(log))]
    out = []
    for prefix, model in API_PREFIX.items():
        parsed = [parse_api_entities(x) for x in log[f"{prefix}_response"]]
        pred = [p or [] for p in parsed]
        lenient = [lenient_sentence_f1(g, p) for g, p in zip(gold, pred)]
        out.append(score(model, "API", gold, pred, lenient, log[f"{prefix}_f1"].to_numpy(),
                         parse_failures=sum(p is None for p in parsed)))
    return out


if __name__ == "__main__":
    conll = load_conll_test()
    df = pd.DataFrame(slm_rows(conll) + api_rows(conll))
    df.to_csv("results/ner_typed_f1.csv", index=False)
    print(df.to_string(index=False))
    print("\nSaved -> results/ner_typed_f1.csv")

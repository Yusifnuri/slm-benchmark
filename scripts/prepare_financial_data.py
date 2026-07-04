"""
Converts the raw Financial PhraseBank release into the CSV format
expected by src/data/dataset_loader.py (columns: 'sentence', 'label').

The raw files under data/FinancialPhraseBank-v1.0/*.txt are:
- "@"-separated: "<sentence>@<sentiment>"
- ISO-8859-1 encoded, CRLF line endings
- sentiment is the string "positive" / "neutral" / "negative"

dataset_loader.py's label_map is {0: "negative", 1: "neutral", 2: "positive"},
so the string labels must be converted to those integer ids before training.

Usage:
    python scripts/prepare_financial_data.py \
        --input data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt \
        --output data/financial_phrasebank.csv
"""

import argparse
import pandas as pd

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


def convert(input_path: str, output_path: str) -> None:
    rows = []
    with open(input_path, encoding="ISO-8859-1", newline="") as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, sentiment = line.rsplit("@", 1)
            sentiment = sentiment.strip().lower()
            if sentiment not in LABEL_TO_ID:
                continue
            rows.append({"sentence": sentence, "label": LABEL_TO_ID[sentiment]})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Wrote {len(df)} rows → {output_path}")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    )
    parser.add_argument("--output", default="data/financial_phrasebank.csv")
    args = parser.parse_args()
    convert(args.input, args.output)

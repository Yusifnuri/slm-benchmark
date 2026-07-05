"""
Dataset loader for all 5 benchmark tasks.

Tasks and datasets (Expose, Section 2, Phase 1):
- classification      → AG News
- ner                 → CoNLL-2003
- summarization       → CNN/DailyMail
- financial_sentiment → Financial PhraseBank (local file)
- code_generation     → MBPP (fine-tuning) / HumanEval (final eval only)

WHY code_generation trains on MBPP instead of HumanEval:
HumanEval ships with only one split ("test", 164 problems) — no train split.
The original version of this loader fine-tuned on that same 164-problem
"test" split, which is also exactly what evaluate_code.py's pass@k benchmark
scores the model on. That means the reported HumanEval numbers would be
measuring memorization of the fine-tuning data, not generalisation — the
model would have seen the literal problems (and solutions) it's later
"tested" on. This invalidates the code_generation column of the benchmark
matrix and would very likely get flagged by any EMNLP/ICLR reviewer.
Fix: fine-tune on MBPP (a different, non-overlapping Python problem set)
and keep HumanEval strictly as an untouched final-eval-only benchmark.
evaluate_code.py already loads HumanEval independently and is unaffected.

WHY classification/financial_sentiment have an explicit train/validation/test
carve-out below instead of just using load_dataset(..., split=split):
AG News ships only "train"/"test" (no "validation"), and the local Financial
PhraseBank CSV has no splits at all — every row is just "the dataset". The
original loader passed split="validation" straight through for both, which
for AG News crashes (unknown split) and for Financial PhraseBank silently
loaded the exact same rows for train, validation-during-training, and final
eval (a prefix-of-train "held-out" set that isn't held out at all — accuracy
numbers would be inflated by training-set leakage). The carve-out below uses
a seeded split so train/validation/test never overlap for either task.
"""

import os
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# Task configuration: dataset names, columns, prompts, label maps
TASK_CONFIGS: Dict[str, Dict] = {
    "classification": {
        # HuggingFace retired the canonical (namespace-less) "ag_news" repo id;
        # this is its current maintained location.
        "dataset": "fancyzhx/ag_news",
        "input_col": "text",
        "label_col": "label",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Technology"},
        "prompt_template": (
            "Classify this news article into exactly one category "
            "(World / Sports / Business / Technology):\n{text}\nCategory:"
        ),
    },
    "ner": {
        # HuggingFace retired the canonical (namespace-less) "conll2003" repo id;
        # this is its current maintained location.
        "dataset": "eriktks/conll2003",
        "revision": "refs/convert/parquet",  # known HuggingFace workaround
        "input_col": "tokens",
        "label_col": "ner_tags",
        # NER tag mapping: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG,
        #                  5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC
        "prompt_template": (
            "Extract named entities (PER=person, ORG=organisation, "
            "LOC=location, MISC=miscellaneous) from this text:\n{text}\nEntities:"
        ),
    },
    "summarization": {
        # HuggingFace retired the canonical (namespace-less) "cnn_dailymail" repo id;
        # this is its current maintained location.
        "dataset": "abisee/cnn_dailymail",
        "config": "3.0.0",
        "input_col": "article",
        "label_col": "highlights",
        "prompt_template": (
            "Summarize the following news article in 2-3 sentences:\n{text}\nSummary:"
        ),
    },
    "financial_sentiment": {
        "dataset": "local",           # requires manual download
        "input_col": "sentence",
        "label_col": "label",
        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
        "prompt_template": (
            "Classify the sentiment of this financial statement "
            "(negative / neutral / positive):\n{text}\nSentiment:"
        ),
    },
    "code_generation": {
        # Fine-tuning source: MBPP, NOT HumanEval — see module docstring.
        # evaluate_code.py loads HumanEval independently for the final,
        # untouched pass@k benchmark; this loader must never hand out
        # HumanEval examples for training.
        "dataset": "google-research-datasets/mbpp",
        "input_col": "text",
        "label_col": "code",
        "prompt_template": "Complete the following Python function:\n{text}\nSolution:",
    },
}


def load_task_dataset(
    task: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    financial_phrasebank_path: Optional[str] = None,
) -> Tuple[HFDataset, Dict]:
    """
    Load the correct dataset for a given task.

    Args:
        task: one of the 5 benchmark tasks
        split: 'train', 'validation', or 'test'
        max_samples: limit dataset size (useful for CPU testing)
        financial_phrasebank_path: absolute path to local CSV file
            Required only for financial_sentiment task.

    Returns:
        (dataset, task_config) tuple
    """
    config = TASK_CONFIGS[task]

    if task == "financial_sentiment":
        # Financial PhraseBank requires manual local download
        if financial_phrasebank_path is None:
            raise ValueError(
                "financial_phrasebank_path is required for financial_sentiment.\n"
                "Run scripts/prepare_financial_data.py first.\n"
                "Expected columns: ['sentence', 'label']"
            )
        df = pd.read_csv(financial_phrasebank_path)
        # Ensure correct column names
        if "sentence" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have 'sentence' and 'label' columns.")
        full = HFDataset.from_pandas(df)

        # No split exists in the source data — carve out a seeded 80/10/10
        # train/validation/test split so the three never overlap.
        trainval = full.train_test_split(test_size=0.2, seed=42)
        val_test = trainval["test"].train_test_split(test_size=0.5, seed=42)
        dataset = {
            "train": trainval["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }[split]

    elif task == "ner":
        # CoNLL-2003 requires parquet revision workaround. trust_remote_code
        # is deliberately omitted: current `datasets` versions serve this
        # revision as parquet (no loading script) and warn that the flag is
        # unsupported/unnecessary here.
        dataset = load_dataset(
            config["dataset"],
            revision=config["revision"],
            split=split,
        )

    elif task == "summarization":
        dataset = load_dataset(
            config["dataset"],
            config["config"],
            split=split,
        )

    elif task == "code_generation":
        # MBPP has real train/validation/test splits — HumanEval is never
        # touched here (see module docstring for why). trust_remote_code
        # omitted; MBPP is served as parquet, no loading script involved.
        dataset = load_dataset(config["dataset"], split=split)

    elif task == "classification":
        if split == "test":
            dataset = load_dataset(config["dataset"], split="test")
        else:
            # AG News ships only train/test — carve a seeded validation
            # slice out of train so periodic eval-during-training never
            # touches the test set used for final benchmark reporting.
            full_train = load_dataset(config["dataset"], split="train")
            train_val = full_train.train_test_split(test_size=0.1, seed=42)
            dataset = train_val["train"] if split == "train" else train_val["test"]

    else:
        raise ValueError(f"Unknown task: {task}")

    # Subsample for fast testing
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    return dataset, config


class SLMDataset(Dataset):
    """
    PyTorch Dataset for all 5 benchmark tasks.

    Converts raw dataset rows into tokenized prompt+completion pairs
    for causal language model fine-tuning.

    The model is trained to predict only the completion (label),
    not the prompt — achieved by masking prompt tokens with -100 in labels.
    """

    def __init__(
        self,
        task: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        financial_phrasebank_path: Optional[str] = None,
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset, self.config = load_task_dataset(
            task, split, max_samples, financial_phrasebank_path
        )
        self.samples = self._preprocess_all()

    def _format_one_sample(self, row: Dict[str, Any]) -> Dict[str, str]:
        """Format a single raw row into prompt + completion strings."""
        config = self.config

        # NER: list of tokens → joined string
        if self.task == "ner":
            text = " ".join(row[config["input_col"]])
        else:
            text = str(row[config["input_col"]])

        # Truncate long inputs to stay within max_length
        text = text[:400]
        prompt = config["prompt_template"].format(text=text)

        # Format completion (label)
        if self.task in ["classification", "financial_sentiment"]:
            completion = config["label_map"][row[config["label_col"]]]
        else:
            completion = str(row[config["label_col"]])

        return {"prompt": prompt, "completion": completion}

    def _preprocess_all(self):
        """Preprocess all samples."""
        return [self._format_one_sample(row) for row in self.dataset]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_text = sample["prompt"] + " " + sample["completion"]

        # Tokenize full text. Padding is left to the data collator (dynamic,
        # per-batch padding) instead of padding every example to max_length here —
        # fixed max_length padding wastes compute on short samples across the batch.
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize prompt only (to find prompt length)
        prompt_enc = self.tokenizer(
            sample["prompt"],
            max_length=self.max_length,
            truncation=True,
        )
        prompt_len = len(prompt_enc["input_ids"])

        # Labels: -100 for prompt tokens (model ignores these in loss)
        # Only completion tokens contribute to training loss
        labels = encoding["input_ids"].squeeze().clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels,
        }
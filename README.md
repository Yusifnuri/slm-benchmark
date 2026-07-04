# slm-benchmark

Fine-Tune or Pay Per Token? An Enterprise Benchmark of Small Language Models.

## Methodology notes (data-leakage guards)

These are deliberate fixes to the original benchmark pipeline. Written down
so the reasoning doesn't have to be reconstructed later while writing up the
thesis methodology section.

**Problem:** the original data loader let the same examples be used for both
fine-tuning and final benchmark evaluation on three of the five tasks. That
inflates the reported accuracy/pass@k numbers and is exactly the kind of
thing a peer reviewer (EMNLP/ICLR Industry Track) would flag.

1. **`code_generation`**: the loader forced *training* data to come from
   HumanEval's only split ("test", 164 problems) — the identical 164
   problems `evaluate_code.py` later scores pass@k on. Fine-tuning on your
   own test set measures memorization, not generalisation.
   → Fixed: fine-tuning now uses **MBPP** (a different, non-overlapping
   Python problem set) for both training and the Trainer's periodic
   validation checks. HumanEval stays completely untouched, loaded only by
   `evaluate_code.py`, reserved solely for the final pass@k benchmark number.

2. **`classification` (AG News)**: AG News only ships `train`/`test` splits —
   there is no `validation` split. The loader used to request `split="validation"`
   anyway, which would crash the very first training run.
   → Fixed: a seeded 90/10 split is carved out of `train` to produce a
   `validation` set for training-time checkpoint selection. The real `test`
   split is never touched until final evaluation.

3. **`financial_sentiment` (Financial PhraseBank)**: this dataset has no
   splits at all — it's a single local CSV. The loader used to load the same
   full file for "train" and "validation" and just slice off the first N
   rows for each, so the "held-out" eval set was actually a subset of the
   training data.
   → Fixed: a seeded 80/10/10 train/validation/test split is carved out once
   in `src/data/dataset_loader.py`, so the three never overlap.

**The resulting convention, consistent across all 5 tasks:**
- `train` → used for fine-tuning (gradient updates).
- `validation` → used only by the `Trainer`'s periodic loss checks /
  `load_best_model_at_end` checkpoint selection during training. Never
  backpropagated through, but also never used for reported numbers.
- `test` → touched exactly once, in `src/evaluation/evaluate.py` (or
  `evaluate_code.py` for code_generation via HumanEval) — these are the
  numbers that go into the benchmark matrix and the thesis.

All splits are seeded (`seed=42`) so they're reproducible across runs.

**Before spending real GPU hours**: `python tests/test_local_cpu.py` exercises
every task's `train`/`validation`/`test` loading path on CPU with 5 samples.
If a dataset id or split name is ever wrong (e.g. an HF Hub repo gets
renamed), this is where it will show up — cheaply, before the full sweep in
`scripts/run_all_experiments.py`.

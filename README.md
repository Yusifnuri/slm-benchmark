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

## Phase 1 (LLM baseline) methodology notes

The `notebooks/02`–`06` baselines call GPT-4o, Claude, and Gemini directly via
API. A few deliberate deviations/fixes from a first-draft version of these
notebooks, so the reasoning survives into the thesis write-up:

1. **Model versions differ from the exposé.** The exposé names
   Claude-3.5-Haiku and Gemini-1.5-Flash; the notebooks actually call
   **`claude-haiku-4-5-20251001`** and **`gemini-2.5-flash`**. This is a
   scope deviation the exposé's own rules say needs supervisor sign-off —
   flag it explicitly in the thesis methodology section, and check whether
   the older models are still reachable via API before deciding this is
   final. `src/evaluation/metrics.py`'s `LLM_API_COSTS` table and
   `results/benchmark_matrix.py`'s `MODELS` list have been updated to match
   the models actually queried, with verified current pricing (Claude Haiku
   4.5: $1.00/$5.00 per 1M input/output tokens; Gemini 2.5 Flash standard
   tier: $0.30/$2.50) — before, the cost table was silently pricing a
   different model than the one being benchmarked, which would have made
   every ROI number in the thesis wrong.
2. **Anthropic calls were missing `temperature=0.0`.** GPT-4o and Gemini were
   already called at `temperature=0.0` (greedy, deterministic) in every
   notebook, but the Claude calls had no `temperature` set — Anthropic's
   default is 1.0. That meant Claude alone had sampling randomness in a
   benchmark meant to compare models under identical decoding conditions.
   Fixed across all 5 notebooks.
3. **Gemini's retry loop was polluting the latency metric.** `summarize_gemini`/
   `classify`/`generate_gemini` retry on a transient 503 with `time.sleep`
   backoff (3s, 6s, 9s...), but `start = time.time()` was set once *before*
   the retry loop — so a retried call's reported "latency" included the
   failed attempts' round-trip time plus the deliberate sleep, sometimes
   ballooning to 100+ seconds for what was really a fast successful call.
   This is why notebook 06's own analysis cell flags a 151s Gemini outlier.
   Fixed by resetting `start` inside each retry attempt, so latency reflects
   only the successful call.
4. **`evaluate_pass_at_1` ran model-generated code via bare `exec()` with no
   timeout**, in both the HumanEval baseline (`06_baseline_humaneval.ipynb`)
   and the SLM fine-tuning eval (`src/evaluation/evaluate_code.py`) — same
   class of risk as any of the other untrusted-code-execution fixes in this
   repo. Fixed to run in a subprocess with a 5-second timeout.
5. **`05_baseline_financial.ipynb` had a hardcoded absolute path**
   (`/Users/yusifnuri/...`) that only works on one machine. Changed to a
   relative path. It already uses `Sentences_AllAgree.txt` (100% annotator
   agreement) — `scripts/prepare_financial_data.py`'s default matches this
   file so the SLM fine-tuning pipeline and the LLM baselines are scored
   against the same ground truth.

6. **No retry or resume on OpenAI/Anthropic failures** — only the Gemini
   calls retried transient errors; a single rate limit or 5xx partway
   through a paid 100-sample loop crashed the whole cell and lost every
   result already paid for. Fixed in all 5 notebooks:
   - `generate_*`/`classify_*`/`extract_entities_*`/`summarize_*` for
     OpenAI and Anthropic now retry up to 3 times with backoff (matching
     the style already used for Gemini), returning an error sentinel
     instead of raising if every attempt fails — one bad sample no longer
     kills the run.
   - Results are checkpointed to a **fixed** (non-timestamped) CSV path
     every 20 samples, and each notebook checks for that file on startup —
     if it exists, already-completed rows are loaded and the loop picks up
     where it left off instead of re-querying (and re-paying for) samples
     already done. Final accuracy/latency/F1/ROUGE aggregates are computed
     from the accumulated `results` list rather than separately-tracked
     running lists, so they're correct whether a run went start-to-finish
     or resumed a partial checkpoint.
   - `06_baseline_humaneval.ipynb` originally contained a superseded
     gpt-4o-mini draft cell alongside the real gpt-4o baseline cell; the
     draft was removed entirely (PR #3), so the notebook now has a single
     baseline cell checkpointing to `logs/humaneval_baseline.csv`.

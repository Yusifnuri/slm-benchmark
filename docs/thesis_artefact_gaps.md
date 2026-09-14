# Thesis-to-repository reconciliation

Compiled 2026-09-10; revised after #24 (validity-audit branch) and again after
the thesis text was brought into line. Every artefact the thesis names must
either exist here or not be claimed. The thesis's fifth contribution is
measurement honesty (§6.2); claiming a released artefact that does not exist
undercuts it directly.

## Resolved by #24

`results/cost_per_request.py`, `results/cost_per_request.csv`,
`results/make_figures.py`, `results/figures/`, and
`scripts/throughput_batch_sweep.py` are on `main`.

`results/benchmark_matrix.csv` has been rebuilt on the re-run data and matches
Table 4.1 of the thesis cell for cell — corrected financial-sentiment sampling,
the full 164-problem HumanEval set, the unified NER instrument,
hardware-consistent H200 pricing, and latency artefacts excluded with the
exclusion count carried as a column.

## Resolved in the thesis text

**`results/predictions/`** does not exist and depends on the pending NER
re-evaluation. Appendix D no longer lists it among the released artefacts; it is
now stated as pending. Appendix F still names it in the reproduction order,
which is correct — `src/evaluation/` writes that directory when it is run.

**Summarisation comparator.** The released matrix puts Gemini 2.5 Flash at
0.223499 ROUGE-L, making it the strongest baseline on that task rather than
GPT-4o at 0.220922. §4.2.3 had carried the superseded 0.219; it now reports
0.223 with the interval [0.210, 0.238] recomputed from
`logs/cnn_dailymail_baseline.csv`. The parity claim in the abstract, §4.7, §5.2
and §6.1 now quotes 0.223, consistent with the strongest-baseline convention the
thesis uses on every other task. Parity is unaffected — Gemini's interval
contains Phi-4-mini's 0.2245.

**Per-request price range.** §5.3 had given the spread across the fine-tuned
models as USD 0.179–0.520 per 1,000 requests on classification. Recomputed at
USD 3.99/GPU-hour and full utilisation from the released latencies — Phi-4-mini
161.77 ms, Llama-3.2-3B 459.50 ms, Mistral-7B 555.93 ms — the range is
USD 0.179–0.616, a factor of 3.4. The lower bound reproduces
`results/cost_per_request.csv` exactly. The argument is unchanged: model choice
remains the smallest of the four factors compared in that sentence.

## Open

Nothing outstanding between the text and this repository. The remaining
pre-submission work is in the thesis document itself (Appendices A, E, F and H).

## Configuration note

`configs/*.yaml` still carry `gpu_cost_per_hour: 2.50` (commented "SRH A6000
approximate USD/hr"). No reported result depends on it — the cost analysis
imputes USD 3.99/hr for the H200 the throughput was measured on (§3.4.3) — but
the field contradicts the thesis at a glance. Appendix B of the thesis carries a
note explaining this. Leave the field as it was at run time so the configs
remain the ones the runs were produced under.

## Hardware note

Every fine-tuning and self-hosted inference run was executed on a single H200
(141 GB) belonging to the university, in Heidelberg, Germany, and made available
to the author at no charge.
No accelerator cost was therefore incurred, and every self-hosted cost, ROI and
breakeven figure in the thesis rests on the imputed commercial rate of USD 3.99
per GPU-hour rather than on an amount paid. The thesis states this in §3.4.3,
in footnote 3 to §3.5.3 and in the Appendix H verification log; the constant
carries the same note at `results/benchmark_matrix.py`, `GPU_COST_PER_HOUR`.
Earlier drafts described the accelerator as rented, which was inaccurate; the
rate and every number derived from it are unchanged.

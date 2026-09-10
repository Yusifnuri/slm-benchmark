# Thesis-to-repository reconciliation

Compiled 2026-09-10, revised after merging #24 (the validity-audit branch).
Every artefact the thesis names must either exist here or be dropped from the
text. The thesis's fifth contribution is measurement honesty (§6.2); claiming a
released artefact that does not exist undercuts it directly.

## Resolved by #24

`results/cost_per_request.py`, `results/cost_per_request.csv`,
`results/make_figures.py`, `results/figures/`, and
`scripts/throughput_batch_sweep.py` are now on `main`.

`results/benchmark_matrix.csv` has been rebuilt on the re-run data and now
matches Table 4.1 of the thesis cell for cell — corrected financial-sentiment
sampling, the full 164-problem HumanEval set, the unified NER instrument,
hardware-consistent H200 pricing, and latency artefacts excluded with the
exclusion count carried as a column.

## Still outstanding

| Referenced as | Referenced in | Status |
|---|---|---|
| `results/predictions/` | Appendix D, §3.6, §4.2.2 | missing — depends on the pending NER re-run |

Until the per-instance predictions exist, Appendix D should not list them among
the released artefacts, and §3.6's claim that the instance-level paired
bootstrap "becomes the primary procedure" stays conditional on that re-run.

## Discrepancy between the thesis text and the released data

`results/benchmark_matrix.csv` gives Gemini 2.5 Flash **0.223499** ROUGE-L on
summarisation. Table 4.1 of the thesis agrees (0.2235). But §4.2.3 of the thesis
reports Gemini at **0.219** — the value from the superseded pre-re-run data.

This matters beyond the one cell. On the released data the strongest
summarisation baseline is Gemini at 0.2235, not GPT-4o at 0.2209, so the
"0.225 versus 0.221" comparison used in the abstract, §4.7, §5.1, §5.2 and §6.1
quotes the second-best baseline. The thesis takes the strongest baseline
everywhere else (0.850 on classification, 0.866 on code generation). The parity
finding itself is unaffected — 0.2245 against 0.2235 is if anything closer
parity — but the quoted comparator needs to be made consistent.

## Configuration note

`configs/*.yaml` still carry `gpu_cost_per_hour: 2.50` (commented "SRH A6000
approximate USD/hr"). No reported result depends on it — the cost analysis
imputes USD 3.99/hr for the H200 the throughput was measured on (§3.4.3) — but
the field contradicts the thesis at a glance. Appendix B of the thesis carries a
note explaining this. Leave the field as it was at run time so the configs
remain the ones the runs were produced under.

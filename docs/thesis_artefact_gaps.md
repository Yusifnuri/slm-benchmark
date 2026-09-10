# Artefacts the thesis claims are released but are not in this repository

Compiled 2026-09-10 while reconciling the thesis text against the repository.
Every item below is referenced by name in the submitted thesis, so each one must
either be added here or removed from the text before submission. The thesis's
fifth contribution is measurement honesty (§6.2); claiming a released artefact
that does not exist undercuts it directly.

## Missing files

| Referenced as | Referenced in | Status |
|---|---|---|
| `results/cost_per_request.csv` | Appendix D | missing |
| `results/cost_per_request.py` | §4.6, Appendix F | missing |
| `results/make_figures.py` | Appendix F | missing |
| `results/predictions/` | Appendix D, §3.6, §4.2.2 | missing |
| `scripts/throughput_batch_sweep.py` | §3.5.3 | missing |

## Stale data

`results/benchmark_matrix.csv` no longer matches Table 4.1 of the thesis, which
states it was generated from this file:

- **Code generation, API arm** — the CSV holds the superseded 50-problem prefix
  scores (GPT-4o 0.52, Claude 0.54, Gemini 0.78). The thesis reports the full
  164-problem run (0.750, 0.793, 0.866). The fine-tuned code-generation cells in
  the CSV *are* current.
- **Financial sentiment, API arm** — the CSV holds the pre-correction
  label-clustered sample (0.99, 0.93, 0.99). The thesis reports the corrected
  common-split run (0.960, 0.940, 0.970).
- **Classification, API arm** — the CSV has GPT-4o at 0.85; the thesis reports
  0.820 with Gemini at 0.850.
- **Latency** — the CSV and Table 4.1 disagree throughout (e.g. classification
  GPT-4o 777.98 ms vs 914 ms).
- **`cost_per_1m_tokens` / `roi_breakeven_tokens`** — still token-denominated
  and computed on the old USD 2.50/hr basis, whereas the thesis moved every
  economic claim onto a per-request basis at USD 3.99/hr (§3.5.3).

## Configuration note

`configs/*.yaml` still carry `gpu_cost_per_hour: 2.50` (commented "SRH A6000
approximate USD/hr"). No reported result depends on it — the cost analysis
imputes USD 3.99/hr for the H200 the throughput was measured on (§3.4.3) — but
the field contradicts the thesis at a glance. Appendix B of the thesis now
carries a note explaining this. Leave the field as it was at run time so the
configs remain the ones the runs were produced under.

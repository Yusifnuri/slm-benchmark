# Thesis correction log — v19 → v32

Every change below was made against evidence in this repository (the MLflow run
history in `mlflow.db`, the released CSVs under `results/`, and the pipeline
source), not against a reading of the prose alone. It exists so that any claim
in the thesis can be traced to the artefact that supports it.

## A. Defects found by auditing the thesis against this repository

| # | Finding | Evidence | Resolution |
|---|---|---|---|
| A1 | The code-generation adapters are fine-tuned on **MBPP**, not HumanEval; HumanEval is evaluation-only. The thesis never said so — Table 3.2 named HumanEval as the dataset. | `src/data/dataset_loader.py` module docstring and `TASK_CONFIGS["code_generation"]`; MLflow records 1,122 samples ÷ 3 epochs = 374 training examples, exactly MBPP's train split. | Disclosed in §3.2.3; Table 3.2 now names both corpora. The code column is labelled a cross-corpus transfer result. |
| A2 | The two arms saw **different configurations of Financial PhraseBank**. The API baselines were scored on AllAgree; the fine-tuned arm was trained and scored on 50Agree. | `scripts/prepare_financial_data.py` takes `Sentences_50Agree.txt`; MLflow records 11,628 ÷ 3 = 3,876 training examples = 80 % of 4,845 (50Agree), not of 2,264 (AllAgree). All 100 API baseline sentences in `logs/financial_baseline_v3_seededsplit.csv` match AllAgree. | Disclosed in §3.2.3, qualified in §3.5.1, flagged as an open comparability defect in §4.2.4 with the direction of its bias stated. **Not silently corrected** — it needs a re-run. |
| A3 | §3.4.2 stated 5,000 training examples per task. True for three tasks only. | MLflow: 5,000 (classification, NER, summarisation), 3,876 (financial sentiment), 374 (code generation). | §3.4.2 now reports the actual per-task counts. |
| A4 | §5.2 claimed financial sentiment was "the only task with N < 5,000" and used that to explain its failure via Condition 2. | Code generation trained on 374 examples — fewer. | §5.2 rewritten: the supervision account is withdrawn for this task, which is now marked as the one cell the proposed mechanism cannot account for. |
| A5 | Figure 4.6 asserted that at u = 0.5 the comparison "stops being decidable". | 0.359 vs 0.551 USD / 1k requests; breakeven ≈ 3,202 requests. `results/make_figures.py` itself notes the band collapses under provider-billed tokens. | Figure and caption now report the breakeven each utilisation implies. |
| A6 | Figure 4.1's legend still described financial sentiment as withdrawn by the validity audit. | `INVALID` in `make_figures.py` is NER-only; §4.2.4 reports the task as re-measured and valid. | Legend corrected. |
| A7 | Figure 4.4 drew nothing for the classification API rows. | Billed rows give `lo == hi`; a zero-length line renders as nothing. | Those rows now render as point markers. |
| A8 | Figure 4.2's legend box covered the Gemini 2.5 Flash row. | — | Legend moved below the axes. |

## B. Document defects corrected

- `(atıf gerekli)` — an untranslated placeholder in §3.4.2 — replaced with a citation to Dettmers et al. (2023).
- Equations (3.1)–(3.5) were absent from §3.5.3 and §3.5.5; the text introduced each with "is given by" and then showed nothing. Inserted and numbered; §4.6's cross-reference repointed from (3.3) to (3.5).
- Four superscript footnote markers had no footnotes behind them. Real Word footnotes inserted.
- Holm (1979), cited in §3.6, was missing from the reference list.
- APA 7: six entries used the 21+-author ellipsis with fewer than 20 authors; all authors now listed. Canonical DOIs added to nine arXiv preprints.
- Figures 4.1–4.4 were never referenced from the running text.
- Table 2.1 was missing from the List of Tables.
- Appendix C carried fine-tuned prompts for C.1–C.2 only; C.3–C.5 added from `dataset_loader.py`. The Sci/Tech vs Technology label-name difference between the arms is now explained.
- Abstract: the 1,700-request breakeven now carries its utilisation and labour assumptions.

## C. Formatting normalised

- `Normal (Web)` (49 % of body paragraphs, a paste-from-browser artefact) folded into `Normal`.
- Body: Times New Roman 12 pt, black, justified, 1.5 line spacing throughout.
- Captions: 10 pt, uniform (previously mixed 9 pt / 10 pt, and mixed within a single caption).
- Table 2.1 was set entirely in bright red (`EE0000`); all nine tables now share one header and zebra style.
- Document default font changed from Calibri to Times New Roman.
- The five `TO BE COMPLETED BEFORE SUBMISSION` notes were left in **red bold** so
  they could not be submitted unnoticed. All five have since been resolved; no
  red text remains in the document.

## D. Second round — artefacts and practitioner validation

- **§3.7.2 / Appendix F**: the decision tool is deployed at
  <https://slm-benchmark-iqwxhymearqtuw7rr6mdmt.streamlit.app/> from the
  `streamlit-deploy` branch. The thesis previously recorded the deployment as
  not realised; exposé Deliverable 1 is now met.
- **Appendix A.1**: five technical prompts added alongside the language-editing
  prompt, covering the NER decode fix, the per-request cost script, the figure
  generation, the MLflow readonly-database bug and the stale-checkpoint fix.
- **§3.7.3 / Appendix E**: all five practitioner sessions planned in the exposé
  were completed, but with five equipment-supply companies in **Azerbaijan**,
  not the Hamburg innovation network the exposé named and the thesis text
  claimed. The count matches the exposé; the recruitment pool does not, and
  that departure is now recorded.
- **Appendix E.1**: the protocol as written said the framework was run during
  each session and its recommendation shown to the participant. It was not —
  the source notes record questions 7–9 as unanswered "because the framework
  has not yet been run with real inputs". The protocol now describes what was
  administered, and states that two of the four framework inputs (volume in
  tokens, minimum accuracy) were assumed rather than elicited.
- **Appendix E.2**: filled from the session notes. The framework was run on the
  five real use cases via `src/decision_framework.py`; it recommends Gemini
  2.5 Flash (API) in all five, unchanged at both ends of every stated volume
  range. Agreement is partial in all five: the deployment mode matches, the
  provider does not.
- **§5.4, §6.1 (SQ5), §6.2**: the validation outcome is now reported. It was
  previously described as having been carried out, with no result given.

## E. Third round — thesis↔repository re-audit (v31 → v32)

Checks that passed and are recorded so they need not be repeated: all 30 cells of
Table 4.1 reproduce `results/benchmark_matrix.csv` exactly (score, latency,
USD/1M tokens and breakeven, including the `none`/`—` distinction); every one of
the 25 repository paths the thesis names exists, the sole exception being
`results/predictions/`, which the thesis itself reports as pending; Appendix E's
five cases, their net promoter scores and their partial-agreement verdicts match
§5.4; the ten constructed scenarios match `results/internal_validation.csv`,
both no-recommendation cases included; the hyperparameters of §3.4.2 match
`configs/*.yaml` (r 16, α 32, dropout 0.05 on the attention projections, 3
epochs, lr 2e-4 cosine, 3 % warmup, effective batch 16, sequence length 512).

| # | Finding | Evidence | Resolution |
|---|---|---|---|
| E1 | §5.4 still described the practitioner base as **three** cases in two sentences, against **five** everywhere else. | Appendix E tabulates five cases with five net promoter scores; §3.7.3 and §6.1 state five. | Both sentences now read five. Five is the correct count; the earlier three-session figure in section D of this log was superseded and has been corrected there. |
| E2 | Figure 4.3's legend named a **Madrid** client; the thesis names **Heidelberg** in §3.3.4, §4.3 and the figure caption. | `results/make_figures.py` carried the only occurrence of "Madrid" anywhere in thesis or repository; the figure embedded in the document reads "Heidelberg". | Legend corrected to Heidelberg and the figure regenerated. |
| E3 | The embedded Figure 4.3 was a stale render — the only one of the six that did not match the pipeline output byte for byte. | MD5 of `word/media/image3.png` against `results/figures/`. | Regenerated after E2 and re-embedded; all six figures now match the released PNGs exactly. |
| E4 | §4.2.4 announced that the majority-class reference quoted below it was **≈0.61, AllAgree's**; the paragraph below quotes **≈0.59**, which is 50Agree's — the split the fine-tuned arm was actually scored on. | §3.5.1 records ≈0.59 on 50Agree and ≈0.61 on AllAgree. | The forward reference now names 0.59 as 50Agree's, with AllAgree's 0.61 given as the baselines' reference. 0.59 was the right figure to quote; only the disclaimer was stale. |
| E5 | §4.2.4 opened by stating that accuracies were reported "using only the AllAgree subset", which holds for the API arm alone. | `scripts/prepare_financial_data.py` takes `Sentences_50Agree.txt` (A2 above). | The opening now states which arm was scored on which configuration. |
| E6 | The abstract implied one outstanding item; two are outstanding. | §4.2.2 withdraws the NER comparison pending a re-run; §4.2.4 leaves the financial-sentiment corpus mismatch open. | The abstract now names both. |
| E7 | §5.4 described every incumbent as "a premium tier", but four of the five buy per-seat chat subscriptions, which are not an API tier — as the same paragraph goes on to say. | Appendix E, cases 1 and 3–5. | Reworded: a premium API in one company, per-seat chat subscriptions in the other four. |
| E8 | §4.6 gave the self-hosted cost at u = 0.5 as USD 0.358 per 1,000 requests. | 0.1793 ÷ 0.5 = 0.3586. A5 of this log already used 0.359. | Corrected to 0.359. The breakeven of ≈3,202 requests is unaffected. |

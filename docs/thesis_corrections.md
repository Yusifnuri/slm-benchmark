# Thesis correction log — v19 → v20

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
- The five `TO BE COMPLETED BEFORE SUBMISSION` notes are deliberately left in **red bold** so they cannot be submitted unnoticed.

## D. Second round — artefacts and practitioner validation

- **§3.7.2 / Appendix F**: the decision tool is deployed at
  <https://slm-benchmark-iqwxhymearqtuw7rr6mdmt.streamlit.app/> from the
  `streamlit-deploy` branch. The thesis previously recorded the deployment as
  not realised; exposé Deliverable 1 is now met.
- **Appendix A.1**: five technical prompts added alongside the language-editing
  prompt, covering the NER decode fix, the per-request cost script, the figure
  generation, the MLflow readonly-database bug and the stale-checkpoint fix.
- **§3.7.3 / Appendix E**: the practitioner sessions were with three
  equipment-supply companies in **Azerbaijan**, not the Hamburg innovation
  network the exposé named and the thesis text claimed. Both departures from
  the exposé (five sessions → three, Hamburg → Azerbaijan) are now recorded.
- **Appendix E.1**: the protocol as written said the framework was run during
  each session and its recommendation shown to the participant. It was not —
  the source notes record questions 7–9 as unanswered "because the framework
  has not yet been run with real inputs". The protocol now describes what was
  administered, and states that two of the four framework inputs (volume in
  tokens, minimum accuracy) were assumed rather than elicited.
- **Appendix E.2**: filled from the session notes. The framework was run on the
  three real use cases via `src/decision_framework.py`; it recommends Gemini
  2.5 Flash (API) in all three, unchanged at both ends of every stated volume
  range. Agreement is partial in all three: the deployment mode matches, the
  provider does not.
- **§5.4, §6.1 (SQ5), §6.2**: the validation outcome is now reported. It was
  previously described as having been carried out, with no result given.

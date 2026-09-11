# Releasing the thesis artefacts

Three exposé deliverables depend on steps that need your own accounts. Everything
that could be prepared in advance has been; what is left is listed here in the
order that costs least time.

## 1. GitHub release tag (2 minutes)

A tag pins the exact code state the thesis was submitted against, which is what
Appendix F asks for.

```bash
git checkout main && git pull
git tag -a v1.0.0-thesis -m "State of the repository at thesis submission"
git push origin v1.0.0-thesis
```

Then paste into **Appendix F**:

> Public repository: https://github.com/Yusifnuri/slm-benchmark, released under
> the MIT licence. The state corresponding to this submission is tagged
> `v1.0.0-thesis`.

## 2. Streamlit Community Cloud — DONE

Deployed at https://slm-benchmark-iqwxhymearqtuw7rr6mdmt.streamlit.app/ from the
`streamlit-deploy` branch. §3.7.2 and Appendix F of the thesis now carry the URL.
The steps below are kept for redeployment.

### Original steps (about 15 minutes)

**The root `requirements.txt` will not work for this.** It carries the full
training stack — torch, transformers, bitsandbytes, mlflow — while `app.py` needs
only `streamlit`, `pandas` and `plotly`. Deploying against it is slow and will
usually fail on the platform's resource limit.

Use the `streamlit-deploy` branch, which is `main` with `requirements.txt`
replaced by the three packages the app actually imports:

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **New app** -> repository `Yusifnuri/slm-benchmark`.
3. Branch: **`streamlit-deploy`** (not `main`).
4. Main file path: `app.py`.
5. Deploy.

Keep the branch in step with `main` when the data changes:

```bash
git checkout streamlit-deploy
git merge main                      # keep this branch's requirements.txt
git checkout streamlit-deploy -- requirements.txt
git commit -am "sync with main, keep slim requirements" && git push
```

Done: §3.7.2 now reads "The application is publicly deployed on Streamlit Community Cloud
at https://slm-benchmark-iqwxhymearqtuw7rr6mdmt.streamlit.app/".

## 3. Hugging Face Hub (about 20 minutes)

All fifteen adapters carry a written model card recording the measured score,
the training configuration, the prompt format the adapter expects, and the
caveats that must travel with the weights — including the three cells that
should not be used (both NER instrument placeholders, Phi-4-mini on financial
sentiment, and Mistral-7B on code generation).

```bash
huggingface-cli login                     # or export HF_TOKEN
python scripts/push_to_hub.py --owner <your-hf-username> --dry-run
python scripts/push_to_hub.py --owner <your-hf-username>
```

Never pass the token as a command-line argument — it ends up in your shell
history. `--private` stages the repos privately if you would rather flip them
public after checking how the cards render.

Then add to **Appendix F**: the fifteen adapter URLs, or the collection URL if
you group them into a Hugging Face collection (faster to cite).

Note: the three `financial_sentiment` adapters are trained on Financial
PhraseBank, which is CC BY-NC-SA 3.0. Their cards state that they are research
artefacts and not commercially deployable. Do not remove that line.

## What this does not cover

Deliverable 4 (the research paper submitted to a peer-reviewed venue by Week 21)
cannot be produced from the repository. Raise it with the first supervisor in
writing and have the outcome recorded, as the exposé's scope-change rule
requires.

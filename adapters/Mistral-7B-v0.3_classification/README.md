---
base_model: mistralai/Mistral-7B-v0.3
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
- lora
- peft
- small-language-model
- enterprise-benchmark
- classification
- qlora
- base_model:adapter:mistralai/Mistral-7B-v0.3
---

# Mistral-7B-v0.3 — Topic classification adapter

A QLoRA (4-bit NF4, double quantisation) adapter that specialises `mistralai/Mistral-7B-v0.3` (7.25 B parameters) for a single enterprise task: it assigns a news item to one of four topics (World, Sports, Business, Sci/Tech).

It was produced for the MSc thesis *Fine-Tune or Pay Per Token? An Enterprise Benchmark of Small Language Models* (SRH University Hamburg), which measures fine-tuned small models against frontier provider APIs on accuracy, latency, cost, privacy exposure and return-on-investment breakeven volume. The adapter is released so that the benchmark can be independently verified.

## Read this before using the adapter

- Adapted with QLoRA from the **base** release, not an instruction-tuned one. Any deficit is jointly attributable to the model and to 4-bit adaptation; the two cannot be separated within this design.

## Measured performance

| Metric | Value |
|---|---|
| Accuracy | **0.69** |
| Mean latency, batch 1 | 556 ms |
| Cost per 1M generated tokens | USD 19.25 |

Measured on a single NVIDIA H200 (141 GB) at batch size one and full utilisation, priced at an imputed USD 3.99 per GPU-hour. Latency excludes network transit. Scores are not comparable across tasks — each task carries its own metric. Evaluation ran on 5 July 2026; the complete matrix is at [`results/benchmark_matrix.csv`](https://github.com/Yusifnuri/slm-benchmark/blob/main/results/benchmark_matrix.csv).

## Training

| | |
|---|---|
| Method | QLoRA (4-bit NF4, double quantisation) |
| Dataset | AG News (`fancyzhx/ag_news`) |
| Dataset licence | Custom, research use |
| Training examples | 5,000 (500 held out for checkpoint selection) |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Epochs | 3 |
| Effective batch size | 16 (2 x 8 gradient accumulation) |
| Max sequence length | 512 tokens |
| Optimiser | AdamW |
| Seed | 42 |

Hyperparameters were held constant across every model and task rather than tuned per cell, so these figures are a conservative lower bound on attainable performance.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", device_map="auto")
model = PeftModel.from_pretrained(base, "<your-hf-username>/Mistral-7B-v0.3_classification")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
```

The adapter was trained on this prompt format and expects it at inference:

```text
Classify the following news text into exactly one category (World / Sports / Business / Technology):
{text}
Category:
```

## Limitations

- Trained once, with a single seed. Reported differences confound model quality with initialisation variance.
- Specialised to one task on one public corpus. It is not a general-purpose assistant and should not be treated as one.
- The evaluation corpora are long-standing public benchmarks and are plausibly present in the base model's pretraining data, which inflates absolute scores.
- Evaluation used 200 held-out instances (all 164 problems for code generation), so detectable effect sizes are bounded at roughly ten percentage points.

## Links

- Code, configurations and evaluation harness: [https://github.com/Yusifnuri/slm-benchmark](https://github.com/Yusifnuri/slm-benchmark)
- Full benchmark matrix: [`results/benchmark_matrix.csv`](https://github.com/Yusifnuri/slm-benchmark/blob/main/results/benchmark_matrix.csv)
- Per-request cost analysis: [`results/cost_per_request.csv`](https://github.com/Yusifnuri/slm-benchmark/blob/main/results/cost_per_request.csv)

## Citation

```bibtex
@mastersthesis{nuri2026finetune,
  title  = {Fine-Tune or Pay Per Token? An Enterprise Benchmark of Small Language Models},
  author = {Nuri, Yusif},
  school = {SRH University Hamburg},
  year   = {2026}
}
```

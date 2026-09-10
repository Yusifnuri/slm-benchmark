---
base_model: meta-llama/Llama-3.2-3B-Instruct
library_name: peft
license: llama3.2
pipeline_tag: text-generation
tags:
- lora
- peft
- small-language-model
- enterprise-benchmark
- classification
- base_model:adapter:meta-llama/Llama-3.2-3B-Instruct
---

# Llama-3.2-3B-Instruct — Topic classification adapter

A LoRA adapter that specialises `meta-llama/Llama-3.2-3B-Instruct` (3.21 B parameters) for a single enterprise task: it assigns a news item to one of four topics (World, Sports, Business, Sci/Tech).

It was produced for the MSc thesis *Fine-Tune or Pay Per Token? An Enterprise Benchmark of Small Language Models* (SRH University Hamburg), which measures fine-tuned small models against frontier provider APIs on accuracy, latency, cost, privacy exposure and return-on-investment breakeven volume. The adapter is released so that the benchmark can be independently verified.

## Read this before using the adapter

- The Llama 3.2 Community Licence permits commercial use but conditions it on attribution, a naming convention for derivative models, and a monthly-active-user eligibility threshold. Check it before adopting.

## Measured performance

| Metric | Value |
|---|---|
| Accuracy | **0.82** |
| Mean latency, batch 1 | 460 ms |
| Cost per 1M generated tokens | USD 15.91 |

Measured on a single NVIDIA H200 (141 GB) at batch size one and full utilisation, priced at an imputed USD 3.99 per GPU-hour. Latency excludes network transit. Scores are not comparable across tasks — each task carries its own metric. Evaluation ran on 5 July 2026; the complete matrix is at [`results/benchmark_matrix.csv`](https://github.com/Yusifnuri/slm-benchmark/blob/main/results/benchmark_matrix.csv).

## Training

| | |
|---|---|
| Method | LoRA |
| Dataset | AG News (`fancyzhx/ag_news`) |
| Dataset licence | Custom, research use |
| Training examples | 5,000 (500 held out for checkpoint selection) |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Epochs | 3 |
| Effective batch size | 16 (4 x 4 gradient accumulation) |
| Max sequence length | 512 tokens |
| Optimiser | AdamW |
| Seed | 42 |

Hyperparameters were held constant across every model and task rather than tuned per cell, so these figures are a conservative lower bound on attainable performance.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "<your-hf-username>/Llama-3.2-3B-Instruct_classification")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
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

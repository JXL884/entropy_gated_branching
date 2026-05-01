# Entropy-Gated Branching for Efficient Test-Time Reasoning

Code for the paper:

> **Entropy-Gated Branching for Efficient Test-Time Reasoning**
> Xianzhi Li, Ethan Callanan, Abdellah Ghassel, Xiaodan Zhu
> *EACL 2026* · [arXiv:2503.21961](https://arxiv.org/abs/2503.21961) · [ACL Anthology](https://aclanthology.org/2026.eacl-long.235)

## Overview

**Entropy-Gated Branching (EGB)** improves test-time reasoning efficiency by branching only at *high-uncertainty* steps, as measured by token-level entropy. At low-entropy (confident) steps, the model continues along a single path; at high-entropy (uncertain) steps, it expands into multiple candidate branches and uses a Process Reward Model (PRM) to prune them. This achieves **22.6% accuracy improvement** over standard inference while being **31–75% faster** than standard beam search on mathematical reasoning benchmarks.
<img width="2041" height="1195" alt="image" src="https://github.com/user-attachments/assets/69168e5e-1d82-47f3-946e-f16ab6e77cc1" />


## Setup

Requires Python 3.11+. We use [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --project pyproject.toml .
```

### Models

Download the required PRM scorer models:

```bash
python download_models.py
```

This fetches `Qwen2.5-Math-PRM-7B` and `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` into `model_cache/`.

Set your HuggingFace cache directory if needed:

```bash
export HF_HOME=/path/to/your/huggingface/cache
```

## Running EGB

```bash
python run_beam_search.py \
  --model Qwen/Qwen3-1.7B \
  --exam gsm \
  --num-beams 4 \
  --num-expansions 4 \
  --entropy-threshold 1.5
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-1.7B` | Generator LLM (any HF causal model) |
| `--scorer-model` | Qwen2.5-Math-PRM-7B | PRM for branch scoring/pruning |
| `--exam` | `gsm` | Dataset: `gsm`, `math`, `aime` |
| `--num-beams` | 4 | Number of beams to maintain |
| `--num-expansions` | 4 | Candidate branches per step |
| `--entropy-threshold` | 1.5 | Entropy threshold for branching |
| `--confidence-beam-search` | off | Use confidence-based variant |
| `--quantize` | off | Enable 4-bit quantization |
| `--limit` | None | Cap number of questions |

## Running Baselines

```bash
python run_baselines.py \
  --baseline self_consistency \
  --model Qwen/Qwen3-1.7B \
  --exam gsm
```

Available baselines: `self_consistency`, `segs`, `diverse_beam`, `conditional_poisson`, `nucleus_best_of_n`.

## Datasets

The `--exam` flag supports the following open-source datasets (downloaded automatically via HuggingFace):

- `gsm` — [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- `math` — [MATH](https://huggingface.co/datasets/lighteval/MATH)
- `aime` — [AIME](https://huggingface.co/datasets/AI-MO/aimo-validation-aime)

> **Note:** The CFA Level 1 and Level 2 exam datasets used in the paper are privately purchased mock exams and are **not publicly available**. Use the open-source datasets above to reproduce the general math reasoning results.

## Slurm

SLURM job scripts for running experiments on a cluster are provided as `run_*.sbatch` files. Adjust resource requests and paths for your cluster environment.

## Citation

```bibtex
@inproceedings{li-etal-2026-entropy,
  title     = {Entropy-Gated Branching for Efficient Test-Time Reasoning},
  author    = {Li, Xianzhi and Callanan, Ethan and Ghassel, Abdellah and Zhu, Xiaodan},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2026.eacl-long.235},
}
```

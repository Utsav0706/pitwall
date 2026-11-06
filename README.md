# Pitwall

[![CI](https://github.com/Utsav0706/pitwall/actions/workflows/ci.yml/badge.svg)](https://github.com/Utsav0706/pitwall/actions/workflows/ci.yml)

LLMOps for one real model: a small open LLM fine-tuned to pick
[Gearbox](https://github.com/Utsav0706/gearbox)'s tier (S/M/L) for a prompt. Pitwall
versions its data, trains it, tracks every run, gates it with
[Dyno](https://github.com/Utsav0706/dyno), serves it, watches it and retrains it.

One of four connected projects: [Scout](https://github.com/Utsav0706/scout) (research
agent), Gearbox (LLM router), Dyno (agent evals) and Pitwall.

**Status:** M1, data. No model results yet.

## Run

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pitwall version
pitwall doctor
```

`pitwall doctor` checks the operations tools (`pip install -e ".[ops]"`) and the local
MLflow and Ollama servers that later steps use.

## Data

Gearbox's labelled prompts (prompt -> cheapest tier that answered well), imported with
DVC from a pinned Gearbox commit, then turned into chat-format pairs whose answer is the
tier:

```bash
pip install -e ".[ops]"
dvc update -R data/gearbox
dvc repro
pitwall data show
```

| File | Rows | S | M | L | What it is |
|---|---|---|---|---|---|
| `test` | 119 | 52 | 57 | 10 | Gearbox's test split; the same for every model |
| `train_old` | 365 | 172 | 180 | 13 | general prompts: the first model learns these |
| `pool_new` | 116 | 37 | 50 | 29 | SimpleQA, AIME, MMLU-Pro: the "new traffic" for the drift demo |
| `train_full` | 481 | 209 | 230 | 42 | old + new: what a retrained model learns |

`dvc.lock` records the hash of every input and output; CI rebuilds the data from scratch
and fails if any hash differs. The instruction text is versioned in `prompts/`
(`pitwall prompts list` shows each template's hash).

## Build steps

| Step | What | Paid cost |
|---|---|---|
| M0 | Skeleton, CI, `pitwall doctor` | £0 |
| M1 | Data from Gearbox under DVC, instruction pairs, splits, prompt registry | £0 |
| M2 | Local MLflow, logistic-regression baseline | £0 |
| M3 | QLoRA fine-tune on free Colab, GGUF Q4 export | £0 |
| M4 | Serve on CPU with Ollama, register and evaluate | £0 |
| M5 | FastAPI `POST /route`, route log, Docker image | £0 |
| M6 | `promote.py`: Dyno gate, under-routing and latency rules; promote / rollback | £0 |
| M7 | Gearbox router plugin, 10% canary | ~£0.15 |
| M8 | Prometheus metrics, Grafana dashboard as code | £0 |
| M9 | Drift report (Evidently) and alert | £0 |
| M10 | Feedback, failures to Dyno cases, retrain loop, drift simulation | ~£0.10 |
| M11 | Docs | £0 |

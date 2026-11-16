# Pitwall

[![CI](https://github.com/Utsav0706/pitwall/actions/workflows/ci.yml/badge.svg)](https://github.com/Utsav0706/pitwall/actions/workflows/ci.yml)

LLMOps for one real model: a small open LLM fine-tuned to pick
[Gearbox](https://github.com/Utsav0706/gearbox)'s tier (S/M/L) for a prompt. Pitwall
versions its data, trains it, tracks every run, gates it with
[Dyno](https://github.com/Utsav0706/dyno), serves it, watches it and retrains it.

One of four connected projects: [Scout](https://github.com/Utsav0706/scout) (research
agent), Gearbox (LLM router), Dyno (agent evals) and Pitwall.

**Status:** M8, metrics, alerts and a Grafana dashboard. Drift detection next (M9).

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

## Baseline and reference rows

All on the same 119 test prompts. *Pass* = the picked tier really answered well (Gearbox's
judge scored it >= 4); *under* = picked a smaller tier than needed (a bad answer); *over*
= a bigger one (wasted money).

| Router | Accuracy | Under | Over | Pass | Picks S/M/L |
|---|---|---|---|---|---|
| always-S | 43.7% | 56.3% | 0.0% | 32.8% | 119/0/0 |
| always-M | 47.9% | 8.4% | 43.7% | 78.2% | 0/119/0 |
| always-L | 8.4% | 0.0% | 91.6% | 84.0% | 0/0/119 |
| Gearbox v1 rules | 45.4% | 42.9% | 11.8% | 45.4% | 87/30/2 |
| Gearbox v2 LLM classifier (Haiku, paid) | 60.5% | 12.6% | 26.9% | 73.9% | 31/82/6 |
| Gearbox v3 embedding + LR | 46.2% | 8.4% | 45.4% | 78.2% | 0/117/2 |
| **LR on `train_old`** (Pitwall's first baseline) | 51.3% | 8.4% | 40.3% | 78.2% | 4/115/0 |
| LR on `train_full` | 46.2% | 8.4% | 45.4% | 78.2% | 0/117/2 |

The LR baseline is reimplemented, not imported from Gearbox; with Gearbox's settings it
gives the same pick as Gearbox v3 on all 119 prompts. Both LRs are close to always-M:
which tier a prompt needs is hard to predict from its text alone. That is the bar the
fine-tuned model has to clear.

```bash
pip install -e ".[ops,baseline]"
python scripts/mlflow_server.py        # MLflow UI at http://127.0.0.1:5000
pitwall reference                       # always-S/M/L + Gearbox's stored predictions
pitwall baseline --train train_old
pitwall runs
```

Every run is tagged with the data version (DVC hash), the prompt template hash and the
git commit.

## Fine-tuning

LoRA on [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) (Apache-2.0,
pinned revision), on a free Colab T4: open
[`notebooks/train_router_colab.ipynb`](notebooks/train_router_colab.ipynb) in Colab and
*Run all* (~20 min). Out comes a bundle: the merged model as GGUF Q4_K_M (~400 MB), the
LoRA adapter, and `bundle.json` (base revision, data version, prompt hash, settings,
validation results, library versions, GGUF sha256). Check it with
`pitwall bundle check bundles/<name>`.

- **Plain LoRA, not QLoRA.** QLoRA keeps the base model in 4 bits so large models fit
  small GPUs; a 0.6B model fits a T4 at full precision, so 4 bits would only add a
  dependency, slower steps and some accuracy loss.
- **No chat template.** The model reads one fixed text (`pitwall/textformat.py`,
  `pitwall-completion/1`) in training and in serving (Ollama raw mode), and answers with
  one token: ` S`, ` M` or ` L`. Its probabilities feed the same cumulative threshold as
  the LR baseline.
- **Tuned on validation only.** Epochs and threshold are picked on a 20% stratified
  hold-out of the training file by routing cost; the 119 test prompts are scored once,
  on the quantised model.
- **Pinned tools.** llama.cpp `b11838`: the converter at a verified commit and a prebuilt
  `llama-quantize` checked against the release's sha256, so nothing is compiled.

**First run (`ft-train_old`, commit f73c05f, Tesla T4, 145 s, peak 3.7 GB GPU):** on the
73 validation prompts the lowest routing cost came from 1 epoch at threshold 0.6, which
picks M for every prompt: the fine-tune, like the LR baseline, found nothing better than
always-M on this data. Epochs 3-4 start separating S from M but under-route more. The
test-set numbers, on the quantised model, come in M4.

`python -m pitwall.finetune.train --smoke ...` runs the whole script (training, choice,
export, bundle) on a tiny random model on CPU in seconds.

## Result so far

On the 119 test prompts, served as Q4 GGUF by Ollama on CPU
([`reports/compare.md`](reports/compare.md)):

| Router | Accuracy | Under | Pass | Quality vs always-L | Cost vs always-L | Routing p50 / p95 |
|---|---|---|---|---|---|---|
| always-M (Haiku) | 47.9% | 8.4% | 78.2% | 94.7% | 6.0% | - |
| Gearbox v2 LLM classifier (paid) | 60.5% | 12.6% | 73.9% | 90.6% | 19.5% | 1952 / 4712 ms |
| Gearbox v3 embedding + LR | 46.2% | 8.4% | 78.2% | 94.7% | 7.1% | 36 / 125 ms |
| LR on `train_old` | 51.3% | 8.4% | 78.2% | 94.7% | 5.9% | - |
| **Fine-tuned Qwen3-0.6B (`ft-train_old`)** | 47.9% | 8.4% | 78.2% | 94.7% | 6.0% | 108 / 488 ms |

The honest reading: **the fine-tune learned the label mix, not the prompts.** Its P(S)
stays between 0.23 and 0.33 for every test prompt, with the same average whatever the
right tier is, so the threshold sends everything to M: it *is* always-M. That matches the
LR and Gearbox's own finding that which tier a prompt needs is barely predictable from
its text. It does replace the paid classifier (v2) at better quality and ~£0 per request,
but not beat a free, simpler rule. The quantised model gives exactly the validation
numbers recorded in training, so this is the model, not the Q4 export.

Latency grows with prompt length on CPU: ~60 ms for short prompts, a median of 315 ms
for prompts over ~350 characters, 1.5 s for the longest.

```bash
pitwall evaluate bundles/ft-train_old   # Ollama on CPU: validation check, test, latency
pitwall register bundles/ft-train_old   # pitwall-router version in the MLflow registry
pitwall compare                          # reports/compare.md (Gearbox's replay for cost)
```

## Routing service

```bash
pip install -e ".[serve]"
python scripts/mlflow_server.py          # registry, port 5000
python scripts/dev_api.py --model v1     # http://127.0.0.1:8003 (Ollama must be running)
curl -X POST http://127.0.0.1:8003/route -H "content-type: application/json"      -d '{"prompt": "Write a haiku about rain"}'
# {"tier": "M", "reason": "ft: P(S)=0.27 P(M)=0.73 P(L)=0.00, threshold 0.6",
#  "model_version": 1, "request_id": "...", "probs": {...}, "latency_ms": 58, "fallback": false}
```

- `model` in the request (or `PITWALL_MODEL`) is an MLflow alias of `pitwall-router`
  (`production`, `candidate`) or a fixed version (`v1`). Aliases are looked up again
  every 30 s, so promoting or rolling back is moving an alias: no restart.
- A version is downloaded from the registry once, checked (bundle check, GGUF hash =
  the registry's tag, prompt template hash = the one it was trained with) and created in
  Ollama over its HTTP API, so the service also works from a container.
- Every decision is written to the route log (`pitwall.db`, Alembic-managed), with the
  prompt as the model saw it: drift detection and retraining read it later. It stays
  local.
- `GET /health` (503 until Ollama answers and the default model is loaded), `GET /models`.
- `Dockerfile`: `python:3.12-slim`, no torch, no model inside; CI builds it.

## Promotion and rollback

The rule is code ([`promote.py`](src/pitwall/promote.py), `decide`): a candidate reaches
production only if

1. **Dyno gate:** on the same 119 test cases, run the same day through the running
   service, its accuracy (exact tier match) drops by no more than `max_drop_pp` (0) from
   production's;
2. its **under-routing** is no higher than production's;
3. its **p95 routing latency** is within `p95_ms_limit` (600 ms, measured in M4).

With no production model yet, the baseline is the trivial rule "always M".

```bash
pitwall candidate v1      # label the version to try (no gate)
pitwall promote           # needs the service running: Dyno calls /route
# pass  dyno_gate      task_success 48% -> 48% (allowed drop 0.0 pts)
# pass  under_routing  v1 8.4% vs always-M 8.4%
# pass  latency_p95    488 ms (limit 600 ms)
# PROMOTED: v1 vs always-M
pitwall rollback          # production back to the version the last promotion replaced
```

Promoting or rolling back moves the `production` alias in the MLflow registry; the
service picks it up within 30 s. Every decision, pass or fail, is an MLflow run
(`kind=promotion` / `rollback`) with the numbers behind it, written *before* the alias
moves. The router answers at temperature 0, so one Dyno attempt per case is enough: two
runs of v1 gave the same tier on all 119 cases.

v1 is production because it matches always-M on every rule, not because it beats it:
it is the starting point the rest of the loop (canary, drift, retraining) improves on.

## Canary through Gearbox

Gearbox's router v4 (`routers/pitwall.py` in Gearbox) asks this service for the tier.
It is registered twice: `v4-finetuned` (Pitwall's `production` model) and
`v4-candidate` (its `candidate`), so Gearbox's own canary setting tries a new version
on a share of real traffic. If Pitwall is down, slow or errors, Gearbox routes to M and
keeps answering.

```bash
pitwall canary config          # runs/gearbox-canary.yaml: Gearbox's config, routed via Pitwall
..\gearbox\.venv\Scripts\python.exe scripts\gearbox_canary.py    # Gearbox on :8002, canary 10%
pitwall canary replay --n 40 --cap-gbp 0.10    # PAID: labelled prompts through Gearbox
pitwall canary report runs/canary-<time>.jsonl # per side: share, tiers, pass, cost, latency
```

First run (both sides are v1, so this tests the mechanism, not a model): 40 requests,
**GBP 0.0082**; `v4-candidate` got 3 (8%), `v4-finetuned` 37, all routed to M. Pass rates
of 33% vs 78% on the *same* model show how little 3 requests say: the canary is evidence
from real traffic, the Dyno gate on all 119 cases makes the decision. With Pitwall
stopped, Gearbox answered both test requests with M (`pitwall unavailable`).

## Monitoring

The service exports Prometheus metrics on `GET /metrics`: decisions by alias, version,
tier and fail-safe; routing latency (a histogram around the 600 ms limit); errors by
reason; which version each alias serves; prompt length. Everything around it is code in
[`monitoring/`](monitoring/):

- **Prometheus** scrape config and four alerts: service down, routing p95 over 600 ms
  for 10 minutes (the promotion limit), fail-safes over 5%, errors.
- **Grafana** datasource and a 10-panel dashboard ("Pitwall router"), provisioned from
  files and read-only, so it can't drift from what is in git.

```bash
python scripts/monitoring.py prometheus   # http://127.0.0.1:9090 (official Windows release)
python scripts/monitoring.py grafana      # http://127.0.0.1:3000
pitwall traffic --n 200                   # free requests straight to /route
```

`monitoring/docker-compose.yml` runs the service with both; CI starts it and checks
that Prometheus scrapes Pitwall, the four rules load and the dashboard exists. A test
checks that every dashboard and alert query uses a metric the service really exports.

There is no cost-per-request panel: Pitwall's decision runs on the local CPU (about
GBP 0). The money is spent by Gearbox's answers, which belong on Gearbox's own metrics.

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

# FAQ

**What is Pitwall?**
The operations side of one model: a 0.6B LLM fine-tuned to choose which of Gearbox's
three model tiers should answer a prompt. Pitwall versions its data, trains it on a
free Colab GPU, tracks and registers each version in MLflow, only promotes a version
that passes a Dyno gate, serves it to Gearbox, monitors it, detects when traffic
changes, and retrains.

**Does the fine-tuned model work?**
It routes every prompt to the medium tier. That beats Gearbox's paid classifier
(94.7% vs 90.6% of the best tier's quality, at a third of its cost) and ties the free
rule "always medium". The tier a prompt needs turned out to be barely predictable from
its text, for every router tried. The README has the numbers.

**So what is the point?**
The loop around the model: reproducible data, a promotion rule written as code, a
canary, rollback in one command, monitoring, drift detection, and failures flowing back
into test cases and training data. All of it ran end to end on real traffic.

**How do I run it?**
Windows, Python 3.12+, Ollama. The README's "Run it" section installs it, rebuilds the
data and starts MLflow and the service. Training needs a Colab GPU
(`notebooks/train_router_colab.ipynb`); everything else runs on a laptop CPU.

**What does it cost?**
Training and serving are free (Colab, CPU). Commands that call paid models (a canary
through Gearbox, sampled judging) print an estimate first and stop at a cap. The whole
project spent about GBP 0.022.

**Can I trust the numbers?**
They are checked against independent sources where possible: Pitwall's metrics
reproduce Gearbox's published router numbers exactly; the rebuilt logistic regression
gives Gearbox v3's pick on all 119 test prompts; Gearbox's replay reproduces its own
report; the quantised model reproduces the validation numbers recorded in training.
The test set is small (119 prompts), and the README says what the numbers can't show.

**What happens to my prompts?**
The service logs each prompt it routes, locally (SQLite), for drift checks and
retraining. Production failures are versioned with DVC and kept out of git. Nothing is
sent anywhere except to Ollama on the same machine; paid commands send prompts to
Gearbox, which sends them to the model it routes to.

**What if Pitwall goes down?**
Gearbox's router v4 falls back to the medium tier and keeps answering; an alert fires
in Prometheus.

**Why not Kubernetes, autoscaling or a GPU server?**
One small model on one machine: an orchestrator, autoscaling and GPU serving would add
cost and moving parts without changing any result.

**Is it deployed?**
Not to the cloud yet. Everything runs locally and in CI, including the service image
and the monitoring stack with Docker Compose. A cloud deployment is planned together
with the other three projects.

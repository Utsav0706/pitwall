"""The `pitwall` command. M0: version, doctor. M1: data build/show, prompts list.
Later milestones add train, register, evaluate, serve, promote, rollback, drift and retrain.

Typer, not argparse: commands and options come from type-hinted functions, so the CLI
and its --help can't drift apart, and CliRunner tests it without a subprocess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from pitwall import __version__
from pitwall import data as data_mod
from pitwall.config import get_settings
from pitwall.doctor import run_checks
from pitwall.log import setup_logging
from pitwall.prompts import PROMPTS_DIR, get_template, load_registry

app = typer.Typer(
    help="Version, fine-tune, gate, serve, monitor and retrain Gearbox's router model.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Training and test data built from Gearbox's dataset.")
prompts_app = typer.Typer(help="Versioned prompt templates.")
app.add_typer(data_app, name="data")
app.add_typer(prompts_app, name="prompts")


@app.callback()
def main() -> None:
    # Windows consoles and pipes use cp1252: a library printing an emoji (MLflow prints
    # one when a run ends) would crash the command. Replace what can't be shown instead.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    setup_logging(get_settings().log_level)


@app.command()
def version() -> None:
    """Print Pitwall's version."""
    typer.echo(f"pitwall {__version__}")


@app.command()
def doctor(
    strict: Annotated[
        bool, typer.Option(help="Exit 1 if any check fails (for scripts and CI).")
    ] = False,
) -> None:
    """Check that the tools and local servers Pitwall uses are installed and running."""
    checks = run_checks(get_settings())
    width = max(len(c.name) for c in checks)
    for check in checks:
        # Plain ASCII: Windows PowerShell 5.1 consoles default to cp1252.
        mark = "ok " if check.ok else "-- "
        typer.echo(f"{mark} {check.name.ljust(width)}  {check.detail}")
    failed = sum(not c.ok for c in checks)
    typer.echo(f"{len(checks) - failed}/{len(checks)} checks passed")
    if strict and failed:
        raise typer.Exit(1)


@data_app.command("build")
def data_build(
    src: Annotated[Path, typer.Option(help="Gearbox files imported by DVC.")] = data_mod.SRC_DIR,
    out: Annotated[Path, typer.Option(help="Where to write the pairs.")] = data_mod.OUT_DIR,
    prompt: Annotated[str, typer.Option(help="Prompt template, name:version.")] = "router:1",
) -> None:
    """Check Gearbox's files and write the instruction pairs (run by `dvc repro`)."""
    try:
        manifest = data_mod.build(src, out, get_template(prompt))
    except (data_mod.DataError, KeyError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(data_mod.table(manifest))


@data_app.command("show")
def data_show(
    out: Annotated[Path, typer.Option(help="Built data folder.")] = data_mod.OUT_DIR,
) -> None:
    """Summarise the built data (from its manifest)."""
    path = out / "manifest.json"
    if not path.exists():
        typer.echo(f"error: {path} not found; run `dvc repro` first", err=True)
        raise typer.Exit(1)
    typer.echo(data_mod.table(json.loads(path.read_text(encoding="utf-8"))))


@prompts_app.command("list")
def prompts_list(
    directory: Annotated[Path, typer.Option(help="Template folder.")] = PROMPTS_DIR,
) -> None:
    """List prompt templates with their hashes."""
    for key, template in load_registry(directory).items():
        typer.echo(f"{key:<16} {template.sha256[:12]}  max {template.max_prompt_chars} chars")


GEARBOX_PREDICTIONS = data_mod.SRC_DIR / "router_predictions.jsonl"


@app.command()
def baseline(
    train: Annotated[
        str, typer.Option(help="Training file in data/derived: train_old or train_full.")
    ] = "train_old",
    c: Annotated[float | None, typer.Option("--c", help="Fixed C (skips the CV search).")] = None,
    class_weight: Annotated[
        str | None, typer.Option(help="Fixed class weight: none or balanced.")
    ] = None,
    threshold: Annotated[float | None, typer.Option(help="Fixed threshold.")] = None,
    name: Annotated[str | None, typer.Option(help="Run name (default lr-<train>).")] = None,
) -> None:
    """Train the embedding + logistic-regression baseline, evaluate, log to MLflow."""
    from pitwall import baseline as lr
    from pitwall import evaluation

    fixed = [c, class_weight, threshold]
    if any(v is not None for v in fixed) and not all(v is not None for v in fixed):
        typer.echo("error: give all of --c, --class-weight and --threshold, or none", err=True)
        raise typer.Exit(1)
    settings = None
    if c is not None:
        weight = None if class_weight in (None, "none") else class_weight
        settings = {"C": c, "class_weight": weight, "threshold": threshold}

    pairs = data_mod.load_pairs(data_mod.OUT_DIR / f"{train}.jsonl")
    test = data_mod.load_pairs(evaluation.TEST_FILE)
    embedder = lr.SentenceTransformerEmbedder()
    trained = lr.train(pairs, embedder, settings)
    router = name or f"lr-{train}"
    result = evaluation.score(router, lr.predict(router, trained.weights, test, embedder), test)
    artifacts = {"weights.json": trained.weights.to_json()}
    if trained.grid:
        artifacts["cv_grid.json"] = {"results": trained.grid}
    run_id = evaluation.log(result, "baseline", {**trained.params, "train_file": train}, artifacts)
    typer.echo(metrics_table([(router, result.metrics)]))
    typer.echo(f"settings: {trained.params}")
    typer.echo(f"mlflow run {run_id}")


@app.command()
def reference(
    predictions: Annotated[
        Path, typer.Option(help="Gearbox's stored router predictions.")
    ] = GEARBOX_PREDICTIONS,
) -> None:
    """Log comparison rows: always-S/M/L and Gearbox's v1, v2 and v3 on the same test set."""
    from pitwall import evaluation

    test = data_mod.load_pairs(evaluation.TEST_FILE)
    rows = []
    for tier in ("S", "M", "L"):
        preds = [evaluation.Prediction(p.id, f"always-{tier}", tier) for p in test]
        rows.append((f"always-{tier}", preds, "pitwall"))
    for router, preds in evaluation.read_predictions(predictions).items():
        rows.append((f"gearbox-{router}", preds, "gearbox"))
    table_rows = []
    for name, preds, source in rows:
        result = evaluation.score(name, preds, test)
        evaluation.log(result, "reference", tags={"source": source})
        table_rows.append((name, result.metrics))
    typer.echo(metrics_table(table_rows))


@app.command()
def runs() -> None:
    """Every run in MLflow's 'router' experiment, newest first."""
    from pitwall.tracking import list_runs

    found = list_runs()
    if not found:
        typer.echo("no runs yet")
        return
    typer.echo(f"{'run':<26} {'kind':<10} {'acc':>6} {'under':>6} {'pass':>6}  data      id")
    for r in found:
        m, t = r["metrics"], r["tags"]

        def pct(key: str, m: dict = m) -> str:
            return f"{100 * m[key]:.1f}%" if key in m else "-"

        typer.echo(
            f"{r['name']:<26} {t.get('kind', '-'):<10} {pct('accuracy'):>6} "
            f"{pct('under_rate'):>6} {pct('pass_rate'):>6}  {t.get('data_version', '-')[:8]:<8}  "
            f"{r['id'][:8]}"
        )


def metrics_table(rows: list[tuple[str, dict]]) -> str:
    from pitwall.metrics import table

    return table(rows)


bundle_app = typer.Typer(help="Model bundles made by a training run.")
app.add_typer(bundle_app, name="bundle")


@bundle_app.command("check")
def bundle_check(
    directory: Annotated[Path, typer.Argument(help="Bundle folder, e.g. bundles/ft-train_old")],
) -> None:
    """Check a bundle is complete and intact (manifest, adapter, GGUF hash)."""
    from pitwall.finetune import bundle

    problems = bundle.check(directory)
    if problems:
        for p in problems:
            typer.echo(f"-- {p}")
        raise typer.Exit(1)
    m = bundle.load_manifest(directory)
    chosen = m.chosen
    typer.echo(f"ok  {m.name}: {m.base_model}, {m.examples} examples, data {m.data_version[:8]}")
    typer.echo(
        f"    chosen on {m.validation_examples} validation prompts: {chosen['epoch']} epochs, "
        f"threshold {chosen['threshold']}, cost {chosen['cost']}, accuracy {chosen['accuracy']}, "
        f"under {chosen['under_rate']}, picks {chosen['picks']}"
    )
    if m.gguf:
        typer.echo(
            f"    {m.gguf.file} {m.gguf.quant} {m.gguf.bytes / 1e6:.0f} MB sha {m.gguf.sha256[:12]}"
        )


@app.command()
def evaluate(
    directory: Annotated[Path, typer.Argument(help="Bundle folder, e.g. bundles/ft-train_old")],
) -> None:
    """Serve a bundle in Ollama on CPU; check it against training, score the test set,
    measure latency, log to MLflow."""
    from pitwall import bundle_eval, evaluation
    from pitwall.config import load_config
    from pitwall.ollama import OllamaRouter, create, model_name

    try:
        manifest = bundle_eval.load_checked(directory)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    settings = get_settings()
    name = model_name(manifest.name)
    typer.echo(f"loading {directory / 'model.gguf'} into Ollama as {name} (CPU)")
    import httpx

    with httpx.Client(base_url=settings.ollama_base_url) as ollama_client:
        create(ollama_client, name, directory / "model.gguf", manifest.gguf.sha256)
    router = OllamaRouter(settings.ollama_base_url, name)
    router.answer("warm-up\n\nTier:")  # loads the model, so it isn't timed

    def progress(i: int, n: int) -> None:
        if i % 20 == 0 or i == n:
            typer.echo(f"  {i}/{n}")

    typer.echo("validation prompts from training (quantisation check):")
    check = bundle_eval.validation_check(router, manifest, Path(manifest.train_file), progress)
    threshold = float(manifest.chosen["threshold"])
    typer.echo(f"test prompts, threshold {threshold}:")
    test = data_mod.load_pairs(evaluation.TEST_FILE)
    scored = bundle_eval.route(router, manifest.name, test, threshold, progress)
    result = evaluation.score(manifest.name, scored.predictions, test)
    latency = bundle_eval.latency_stats(scored.latencies)
    result.metrics.update(latency)
    result.metrics["no_answer"] = scored.no_answer
    for side in ("trained", "gguf"):
        for k in ("cost", "accuracy", "under_rate"):
            result.metrics[f"val_{side}_{k}"] = check[side][k]
    run_id = evaluation.log(
        result,
        "finetune",
        bundle_eval.manifest_params(manifest),
        {"bundle.json": manifest.model_dump(), "validation_check.json": check},
        bundle_eval.manifest_tags(manifest),
    )
    limit = load_config().promotion.p95_ms_limit
    typer.echo("")
    typer.echo(metrics_table([(manifest.name, result.metrics)]))
    typer.echo(
        f"validation: trained cost {check['trained']['cost']} acc {check['trained']['accuracy']} "
        f"picks {check['trained']['picks']} | GGUF cost {check['gguf']['cost']} "
        f"acc {check['gguf']['accuracy']} picks {check['gguf']['picks']}"
    )
    typer.echo(
        f"latency p50 {latency['latency_p50_ms']} ms, p95 {latency['latency_p95_ms']} ms "
        f"(limit {limit} ms: {'ok' if latency['latency_p95_ms'] <= limit else 'OVER'}), "
        f"max {latency['latency_max_ms']} ms; no_answer {scored.no_answer}"
    )
    typer.echo(f"mlflow run {run_id}")


@app.command()
def register(
    directory: Annotated[Path, typer.Argument(help="Evaluated bundle folder")],
) -> None:
    """Add an evaluated bundle to the MLflow registry as a new pitwall-router version."""
    from pitwall import bundle_eval, registry

    try:
        bundle_eval.load_checked(directory)
        version, created = registry.register(directory)
    except (ValueError, registry.RegistryError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    state = "registered" if created else "already registered"
    typer.echo(f"{state}: {registry.REGISTERED_MODEL} version {version.version} ({directory.name})")


@app.command()
def compare(
    out: Annotated[Path, typer.Option(help="Report file.")] = Path("reports/compare.md"),
) -> None:
    """Write the comparison of every router (routing metrics + Gearbox's cost replay)."""
    import mlflow

    from pitwall import compare as cmp

    runs = cmp.latest_runs()
    if not runs:
        typer.echo("no runs yet", err=True)
        raise typer.Exit(1)
    ours = {}
    for r in runs:
        if r["tags"]["kind"] in ("baseline", "finetune"):
            path = mlflow.artifacts.download_artifacts(
                run_id=r["id"], artifact_path="predictions.jsonl"
            )
            ours[r["name"]] = Path(path).read_text(encoding="utf-8")
    replay = cmp.gearbox_replay(get_settings().gearbox_dir, GEARBOX_PREDICTIONS, ours)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cmp.render(runs, replay), encoding="utf-8", newline="\n")
    typer.echo(f"wrote {out} ({len(runs)} routers, {len(replay)} replay rows)")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to listen on.")] = "127.0.0.1",
    port: Annotated[int | None, typer.Option(help="Port (default PITWALL_PORT, 8003).")] = None,
) -> None:
    """Run the routing service: POST /route, GET /health, GET /models."""
    import uvicorn

    from pitwall.serving.app import create_app

    uvicorn.run(create_app(), host=host, port=port or get_settings().pitwall_port)


@data_app.command("dyno-suite")
def data_dyno_suite() -> None:
    """Write the promotion gate's Dyno suite from the test pairs (run by `dvc repro`)."""
    from pitwall.dyno_suite import SUITE_FILE, write_suite

    typer.echo(f"wrote {SUITE_FILE} ({write_suite()} cases)")


@app.command()
def candidate(
    ref: Annotated[str, typer.Argument(help="Registered version to try, e.g. v2")],
) -> None:
    """Point the 'candidate' alias at a registered version (no gate: it is only a label)."""
    from pitwall import promote as p

    try:
        version = p.set_candidate(ref)
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"candidate -> v{version}")


@app.command()
def promote(
    ref: Annotated[
        str, typer.Argument(help="Version to promote: candidate (alias) or vN")
    ] = "candidate",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Run the gate and log it; don't move the alias.")
    ] = False,
) -> None:
    """Gate a candidate against production (Dyno + under-routing + p95); promote on a pass.
    Needs the service running (pitwall serve) so Dyno can call /route."""
    from pitwall import promote as p
    from pitwall.config import load_config

    try:
        result = p.promote(ref, get_settings(), load_config().promotion, dry_run)
    except p.PromotionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for check in result.decision.checks:
        typer.echo(f"{'pass' if check.passed else 'FAIL'}  {check.name:<14} {check.detail}")
    if result.decision.promote:
        verdict = "WOULD PROMOTE (dry run)" if dry_run else "PROMOTED"
    else:
        verdict = "REJECTED"
    typer.echo(
        f"{verdict}: v{result.candidate_version} vs {result.baseline} "
        f"(dyno runs {result.dyno_runs['candidate']} / {result.dyno_runs['baseline']}, "
        f"mlflow run {result.mlflow_run[:8]})"
    )
    if not result.decision.promote:
        raise typer.Exit(1)


@app.command()
def rollback() -> None:
    """Put production back on the version the last promotion replaced."""
    from pitwall import promote as p

    try:
        old, new, run = p.rollback()
    except p.PromotionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"rolled back: production {old} -> {new} (mlflow run {run[:8]})")


canary_app = typer.Typer(help="Canary release through Gearbox (production + candidate).")
app.add_typer(canary_app, name="canary")


@canary_app.command("config")
def canary_config(
    percent: Annotated[
        float | None, typer.Option(help="Candidate share (default config.yaml).")
    ] = None,
) -> None:
    """Write runs/gearbox-canary.yaml: Gearbox's config routed through Pitwall."""
    from pitwall import canary as c
    from pitwall.config import load_config

    settings = get_settings()
    share = percent if percent is not None else load_config().canary.percent
    path = c.write_config(settings.gearbox_dir, share, settings.pitwall_url)
    typer.echo(f"wrote {path} (router {c.PRODUCTION_ROUTER}, canary {c.CANDIDATE_ROUTER} {share}%)")


@canary_app.command("replay")
def canary_replay(
    n: Annotated[int, typer.Option(help="Prompts to send.")] = 40,
    cap_gbp: Annotated[float, typer.Option(help="Stop before spending more than this.")] = 0.10,
    seed: Annotated[int, typer.Option(help="Which prompts, in which order.")] = 7,
    out: Annotated[
        Path | None, typer.Option(help="Rows file (default runs/canary-<time>.jsonl).")
    ] = None,
) -> None:
    """Send labelled test prompts through Gearbox (PAID: Gearbox calls its model tiers)."""
    import time as _time

    import httpx

    from pitwall import canary as c
    from pitwall import evaluation

    prompts = c.pick_prompts(data_mod.load_pairs(evaluation.TEST_FILE), n, seed)
    out = out or Path(f"runs/canary-{_time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    typer.echo(
        f"replaying {len(prompts)} prompts through {get_settings().gearbox_url}, cap GBP {cap_gbp}"
    )

    def progress(i: int, total: int, spent: float) -> None:
        if i % 10 == 0 or i == total:
            typer.echo(f"  {i}/{total}  GBP {spent:.4f}")

    with httpx.Client(base_url=get_settings().gearbox_url, timeout=180) as client:
        sent, spent, capped = c.replay(client, prompts, cap_gbp, out, progress)
    typer.echo(
        f"{sent} requests, GBP {spent:.4f}{' (stopped at the cap)' if capped else ''} -> {out}"
    )


@canary_app.command("report")
def canary_report(
    rows_file: Annotated[Path, typer.Argument(help="A runs/canary-*.jsonl file")],
) -> None:
    """Compare production and candidate on a replay; log it to MLflow (kind=canary)."""
    import mlflow

    from pitwall import canary as c
    from pitwall import evaluation, metrics, tracking

    rows = c.read_rows(rows_file)
    summary = c.summarise(rows, metrics.load_scores(evaluation.SCORES_FILE))
    errors = sum(r["status"] != 200 for r in rows)
    typer.echo(c.render(summary, errors))
    with tracking.run(
        f"canary {rows_file.stem}", "canary", tags={"rows_file": rows_file.name}
    ) as r:
        for router, s in summary.items():
            side = "candidate" if router == c.CANDIDATE_ROUTER else "production"
            for key in (
                "requests",
                "share",
                "pass_rate",
                "cost_per_request_gbp",
                "latency_p95_ms",
                "fail_safe",
            ):
                if s[key] is not None:
                    mlflow.log_metric(f"{side}.{key}", float(s[key]))
        mlflow.log_dict({"summary": summary, "errors": errors}, "canary.json")
        typer.echo(f"mlflow run {r.info.run_id[:8]}")


@app.command()
def traffic(
    source: Annotated[
        str, typer.Option(help="Prompts from data/derived: test, train_old or pool_new (drift).")
    ] = "test",
    n: Annotated[int, typer.Option(help="Requests to send (prompts repeat if fewer).")] = 100,
    model: Annotated[str | None, typer.Option(help="Model reference (default: service's).")] = None,
    seed: Annotated[int, typer.Option(help="Prompt order.")] = 7,
    part: Annotated[
        str | None, typer.Option(help="Only prompts of this part: old or hard.")
    ] = None,
) -> None:
    """Send prompts straight to Pitwall's /route (free: local CPU), e.g. to fill the
    dashboards, or `--source pool_new` to make the traffic drift on purpose (M9)."""
    import random as _random
    from collections import Counter

    import httpx

    pairs = data_mod.load_pairs(data_mod.OUT_DIR / f"{source}.jsonl")
    if part:
        pairs = [p for p in pairs if p.part == part]
    if not pairs:
        typer.echo(f"error: no prompts in {source} with part={part}", err=True)
        raise typer.Exit(1)
    texts = [data_mod.prompt_text(p) for p in pairs]
    rng = _random.Random(seed)
    tiers: Counter[str] = Counter()
    errors = 0
    with httpx.Client(base_url=get_settings().pitwall_url, timeout=60) as client:
        for i in range(1, n + 1):
            body = {"prompt": rng.choice(texts)}
            if model:
                body["model"] = model
            r = client.post(
                "/route", json=body, headers={"x-request-id": f"traffic-{source}-{seed}-{i}"}
            )
            if r.status_code == 200:
                tiers[r.json()["tier"]] += 1
            else:
                errors += 1
            if i % 50 == 0 or i == n:
                typer.echo(f"  {i}/{n}  tiers {dict(tiers)}  errors {errors}")


drift_app = typer.Typer(help="Drift: are recent prompts still like the training data?")
app.add_typer(drift_app, name="drift")


@drift_app.command("check")
def drift_check(
    window: Annotated[int, typer.Option(help="The last N prompts the model routed.")] = 200,
    model: Annotated[
        str, typer.Option(help="Model reference whose traffic to check.")
    ] = "production",
) -> None:
    """Compare recent prompts with the model's training prompts (length, task mix,
    embeddings). Exit code 3 when drift is detected, so a script can act on it."""
    import mlflow

    from pitwall import baseline, drift, tracking
    from pitwall.config import load_config
    from pitwall.finetune import bundle as bundle_mod
    from pitwall.serving.db import open_engine
    from pitwall.serving.models import MlflowResolver, ModelUnavailable

    settings = get_settings()
    resolver = MlflowResolver()
    try:
        resolved = resolver.resolve(model)
    except ModelUnavailable as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    directory = settings.model_cache_dir / f"v{resolved.version}" / "bundle"
    if not (directory / bundle_mod.MANIFEST).exists():
        directory = resolver.fetch(resolved, settings.model_cache_dir / f"v{resolved.version}")
    manifest = bundle_mod.load_manifest(directory)
    train_file = Path(manifest.train_file)
    if data_mod.sha256_file(train_file) != manifest.train_sha256:
        typer.echo(
            f"error: {train_file} is not the file v{resolved.version} was trained on", err=True
        )
        raise typer.Exit(1)

    engine = open_engine(settings.database_url)
    logged = drift.current_window(engine, model, window)
    current = drift.distinct(logged)
    typer.echo(f"window: last {len(logged)} {model} requests, {len(current)} distinct prompts")
    reference = data_mod.load_pairs(train_file)
    ref_texts = [data_mod.prompt_text(p) for p in reference]
    embedder = baseline.SentenceTransformerEmbedder()
    try:
        result = drift.check(
            ref_texts,
            baseline.embed_cached(ref_texts, embedder),
            [p.task_type for p in reference],
            current,
            embedder.encode(current),
            load_config().drift.threshold,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"v{resolved.version} ({manifest.name}): {train_file.name} "
        f"vs {len(current)} distinct recent {model} prompts"
    )
    typer.echo(drift.render(result))

    html = drift.report_path()
    html.parent.mkdir(parents=True, exist_ok=True)
    result.snapshot.save_html(str(html))
    reference_label = f"v{resolved.version} {train_file.name}"
    drift.save_record(engine, result, model, reference_label)
    with tracking.run(f"drift {model}", "drift", tags={"reference": reference_label}) as r:
        mlflow.log_metrics(
            {"share": result.share, "detected": float(result.detected)}
            | {f"{f.name}.value": f.value for f in result.features}
            | {f"{f.name}.drifted": float(f.drifted) for f in result.features}
        )
        mlflow.log_dict(result.to_record(), "drift.json")
        mlflow.log_artifact(str(html))
    typer.echo(f"report {html}; mlflow run {r.info.run_id[:8]}")
    if result.detected:
        raise typer.Exit(3)


feedback_app = typer.Typer(help="Thumbs up / down on routing decisions.")
app.add_typer(feedback_app, name="feedback")


@feedback_app.command("send")
def feedback_send(
    request_id: Annotated[str, typer.Argument(help="Pitwall request id (from /route)")],
    rating: Annotated[str, typer.Argument(help="up or down")],
    comment: Annotated[str | None, typer.Option(help="Why.")] = None,
) -> None:
    """POST /feedback to the running service."""
    import httpx

    body = {"request_id": request_id, "rating": rating, "comment": comment}
    r = httpx.post(f"{get_settings().pitwall_url}/feedback", json=body, timeout=10)
    if r.status_code != 201:
        typer.echo(f"error: {r.status_code} {r.text[:300]}", err=True)
        raise typer.Exit(1)
    typer.echo(f"feedback {r.json()['id']}: {rating} on {request_id}")


@app.command("sample-eval")
def sample_eval(
    n: Annotated[int, typer.Option(help="Recent production prompts to judge.")] = 10,
    cap_gbp: Annotated[float, typer.Option(help="Refuse above this estimate.")] = 0.10,
    yes: Annotated[bool, typer.Option("--yes", help="Actually run it (PAID).")] = False,
) -> None:
    """Answer a sample of recent production prompts through Gearbox and judge them with
    Dyno's quality judge; failing answers become failure rows (PAID: Gearbox + judge)."""
    import time as _time

    from pitwall import loop
    from pitwall.serving.db import open_engine

    settings = get_settings()
    prompts = loop.sample_prompts(open_engine(settings.database_url), n)
    estimate = loop.estimate_gbp(len(prompts))
    typer.echo(f"{len(prompts)} prompts, estimate GBP {estimate:.3f} (cap {cap_gbp:.2f})")
    if estimate > cap_gbp:
        typer.echo("REFUSED: estimate over the cap", err=True)
        raise typer.Exit(1)
    if not yes:
        typer.echo("dry run: add --yes to send it")
        return
    stamp = _time.strftime("%Y%m%d-%H%M%S")
    suite = Path(f"runs/sample-{stamp}.yaml")
    cases = loop.write_sample_suite(prompts, suite)
    try:
        run_id, spent = loop.run_sample(suite, settings, cap_gbp)
        rows, report = loop.dyno_failures(run_id, cases, settings)
    except loop.LoopError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    out = Path(f"runs/failures-sample-{stamp}.jsonl")
    loop.write_failures(rows, out)
    judge = report.get("judge_cost_gbp", 0.0)
    typer.echo(
        f"dyno run {run_id}: {len(cases) - len(rows)}/{len(cases)} answers passed the quality "
        f"judge; GBP {spent:.4f} answers + {judge:.4f} judge -> {len(rows)} failures in {out}"
    )


failures_app = typer.Typer(help="Production failures -> Dyno cases + training queue.")
app.add_typer(failures_app, name="failures")


@failures_app.command("import")
def failures_import(
    samples: Annotated[
        list[Path] | None, typer.Argument(help="runs/failures-sample-*.jsonl files to include")
    ] = None,
) -> None:
    """Thumbs-downs (from the database) + sample-eval failures -> new Dyno cases
    (dyno import-failures) + the training queue, versioned with DVC."""
    import time as _time

    from pitwall import loop
    from pitwall.serving.db import open_engine

    rows = loop.feedback_failures(open_engine(get_settings().database_url))
    for path in samples or []:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(loop.FailureRow(**json.loads(line)))
    if not rows:
        typer.echo("no failures to import")
        return
    stamp = _time.strftime("%Y%m%d-%H%M%S")
    combined = Path(f"runs/failures-{stamp}.jsonl")
    loop.write_failures(rows, combined)
    try:
        cases_file, message = loop.import_cases(combined, stamp)
        queued = loop.queue_for_labelling(rows)
        version = loop.dvc_add()
    except loop.LoopError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{len(rows)} failures ({combined})")
    typer.echo(f"dyno: {message}")
    typer.echo(f"training queue: +{queued} prompts in {loop.QUEUE}")
    typer.echo(f"data/production is now DVC version {version[:12]} (commit data/production.dvc)")


retrain_app = typer.Typer(help="New data -> Colab training -> gate -> canary -> promote.")
app.add_typer(retrain_app, name="retrain")


@retrain_app.command("prepare")
def retrain_prepare(
    train: Annotated[str, typer.Option(help="Training file in data/derived.")] = "train_full",
) -> None:
    """Check the training data and print the Colab steps."""
    from pitwall import retrain

    path = data_mod.OUT_DIR / f"{train}.jsonl"
    if not path.exists():
        typer.echo(f"error: {path} not found; run `dvc repro`", err=True)
        raise typer.Exit(1)
    pairs = data_mod.load_pairs(path)
    from collections import Counter

    labels = Counter(p.label for p in pairs)
    parts = Counter(p.part for p in pairs)
    typer.echo(f"{path}: {len(pairs)} prompts, labels {dict(labels)}, parts {dict(parts)}")
    typer.echo(f"sha256 {data_mod.sha256_file(path)[:12]} (the bundle records it)")
    typer.echo("\nOn Colab (free T4, ~20 min):")
    typer.echo(f"  1. open {retrain.COLAB_URL}")
    typer.echo(f'  2. first code cell: TRAIN = "{train}"')
    typer.echo("  3. Runtime -> Change runtime type -> T4 GPU, then Run all")
    typer.echo(f"  4. unzip ft-{train}.zip into bundles/")
    typer.echo(f"\nThen: pitwall retrain finish bundles/ft-{train}")


@retrain_app.command("finish")
def retrain_finish(
    bundle: Annotated[Path, typer.Argument(help="The new bundle, e.g. bundles/ft-train_full")],
    canary_n: Annotated[
        int, typer.Option(help="Canary requests through Gearbox (PAID; 0 = skip).")
    ] = 40,
    cap_gbp: Annotated[float, typer.Option(help="Canary spending cap.")] = 0.10,
    yes: Annotated[bool, typer.Option("--yes", help="Allow the paid canary.")] = False,
) -> None:
    """Evaluate -> register -> candidate -> gate (dry run) -> canary -> promote.
    Needs the service running; the canary also needs Gearbox in canary mode."""
    from pitwall import retrain

    if canary_n > 0 and not yes:
        typer.echo(
            f"the canary sends {canary_n} requests through Gearbox (PAID, cap GBP "
            f"{cap_gbp}); add --yes, or --canary-n 0 to skip it",
            err=True,
        )
        raise typer.Exit(1)
    try:
        version = retrain.finish(bundle, canary_n, cap_gbp, typer.echo)
    except retrain.RetrainError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"\n{version} is production")

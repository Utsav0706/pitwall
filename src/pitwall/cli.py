"""The `pitwall` command. M0: version, doctor. M1: data build/show, prompts list.
Later milestones add train, register, evaluate, serve, promote, rollback, drift and retrain.

Typer, not argparse: commands and options come from type-hinted functions, so the CLI
and its --help can't drift apart, and CliRunner tests it without a subprocess.
"""

from __future__ import annotations

import json
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
    create(name, directory / "model.gguf")
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

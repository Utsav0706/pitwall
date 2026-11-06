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
    typer.echo(f"{'run':<22} {'kind':<10} {'acc':>6} {'under':>6} {'pass':>6}  data      id")
    for r in found:
        m, t = r["metrics"], r["tags"]

        def pct(key: str, m: dict = m) -> str:
            return f"{100 * m[key]:.1f}%" if key in m else "-"

        typer.echo(
            f"{r['name']:<22} {t.get('kind', '-'):<10} {pct('accuracy'):>6} "
            f"{pct('under_rate'):>6} {pct('pass_rate'):>6}  {t.get('data_version', '-')[:8]:<8}  "
            f"{r['id'][:8]}"
        )


def metrics_table(rows: list[tuple[str, dict]]) -> str:
    from pitwall.metrics import table

    return table(rows)

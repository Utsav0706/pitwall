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

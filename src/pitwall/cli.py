"""The `pitwall` command. M0: version, doctor. Later milestones add data, train,
register, evaluate, serve, promote, rollback, drift and retrain.

Typer, not argparse: commands and options come from type-hinted functions, so the CLI
and its --help can't drift apart, and CliRunner tests it without a subprocess.
"""

from __future__ import annotations

from typing import Annotated

import typer

from pitwall import __version__
from pitwall.config import get_settings
from pitwall.doctor import run_checks
from pitwall.log import setup_logging

app = typer.Typer(
    help="Version, fine-tune, gate, serve, monitor and retrain Gearbox's router model.",
    no_args_is_help=True,
)


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

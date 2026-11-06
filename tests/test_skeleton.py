from __future__ import annotations

import json
import logging
from importlib import metadata
from io import StringIO
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from pitwall import __version__, doctor
from pitwall.cli import app
from pitwall.config import Settings, load_config
from pitwall.log import setup_logging

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_repo_config_is_valid() -> None:
    config = load_config(ROOT / "config.yaml")
    assert config.canary.percent == 10
    assert config.promotion.p95_ms_limit > 0


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    bad.write_text(text + "\ncanery:\n  percent: 5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_config_rejects_out_of_range(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    text = (ROOT / "config.yaml").read_text(encoding="utf-8").replace("percent: 10", "percent: 150")
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_settings_defaults_ignore_blank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEARBOX_URL", "")
    settings = Settings(_env_file=None)
    assert settings.gearbox_url == "http://127.0.0.1:8002"
    assert settings.pitwall_port == 8003


def test_missing_package_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_version(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(doctor.metadata, "version", fake_version)
    check = doctor.check_package("mlflow", "M2")
    assert not check.ok
    assert "M2" in check.detail


def test_package_that_fails_to_import_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.metadata, "version", lambda name: "0.7.23")

    def broken_import(name: str) -> None:
        raise ImportError("cannot import name 'infer_dtype_from_object'")

    monkeypatch.setattr(doctor.importlib, "import_module", broken_import)
    check = doctor.check_package("evidently", "M9")
    assert not check.ok
    assert "import fails" in check.detail


def test_installed_package_passes() -> None:
    check = doctor.check_package("typer", "M0")
    assert check.ok, check.detail


def test_unreachable_server_is_reported() -> None:
    # Port 9 (discard) is closed on normal machines: the check must fail fast, not raise.
    check = doctor.check_url("ollama", "http://127.0.0.1:9/api/version", timeout=0.5)
    assert not check.ok
    assert "not reachable" in check.detail


def test_doctor_strict_exits_1_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pitwall.cli.run_checks", lambda settings: [doctor.Check("x", False, "missing")]
    )
    assert runner.invoke(app, ["doctor"]).exit_code == 0
    result = runner.invoke(app, ["doctor", "--strict"])
    assert result.exit_code == 1
    assert "0/1 checks passed" in result.output


def test_logs_are_json_with_extra_fields() -> None:
    stream = StringIO()
    setup_logging("INFO", stream)
    logging.getLogger("pitwall.test").info("hello", extra={"model_version": 3})
    line = json.loads(stream.getvalue())
    assert line["msg"] == "hello"
    assert line["model_version"] == 3

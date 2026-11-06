from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pitwall import data
from pitwall.cli import app
from pitwall.prompts import PromptTemplate, get_template, load_registry, load_template

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = PromptTemplate(
    name="router",
    version=1,
    system="Pick S, M or L.",
    user_template="{prompt}",
    max_prompt_chars=10,
)

PROMPTS = [
    {"id": "dolly-1", "prompt": "Write a haiku", "task_type": "writing", "source": "dolly"},
    {"id": "gsm8k-1", "prompt": "2+2?", "task_type": "reasoning", "source": "gsm8k"},
    {"id": "simpleqa-1", "prompt": "Who won X?", "task_type": "research", "source": "simpleqa"},
    {"id": "aime2025-1", "prompt": "Find n.", "task_type": "reasoning", "source": "aime2025"},
    {"id": "mbpp-1", "prompt": "Sort a list", "task_type": "coding", "source": "mbpp"},
]
LABELS = {"dolly-1": "S", "gsm8k-1": "S", "simpleqa-1": "L", "aime2025-1": "M", "mbpp-1": "M"}
SPLIT = {"train": ["dolly-1", "gsm8k-1", "simpleqa-1", "aime2025-1"], "test": ["mbpp-1"]}


def write_src(src: Path, prompts=PROMPTS, labels=LABELS, split=SPLIT, sha: str | None = None):
    src.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(p) + "\n" for p in prompts).encode("utf-8")
    (src / "prompts.jsonl").write_bytes(body)
    (src / "labels.jsonl").write_text(
        "".join(json.dumps({"id": i, "label": t}) + "\n" for i, t in labels.items()),
        encoding="utf-8",
    )
    (src / "split.json").write_text(json.dumps(split), encoding="utf-8")
    digest = sha or hashlib.sha256(body).hexdigest()
    (src / "manifest.json").write_text(json.dumps({"prompts_sha256": digest}), encoding="utf-8")
    return src


def test_build_splits_old_and_hard(tmp_path: Path) -> None:
    src = write_src(tmp_path / "src")
    manifest = data.build(src, tmp_path / "out", TEMPLATE)
    files = manifest["files"]
    assert files["test"]["rows"] == 1
    assert files["train_old"]["rows"] == 2  # dolly, gsm8k
    assert files["pool_new"]["rows"] == 2  # simpleqa, aime
    assert files["train_full"]["rows"] == 4
    assert files["pool_new"]["by_label"] == {"L": 1, "M": 1}
    pool_ids = [r["id"] for r in data.read_jsonl(tmp_path / "out" / "pool_new.jsonl")]
    assert pool_ids == ["aime2025-1", "simpleqa-1"]  # sorted by id


def test_pair_format_and_cut(tmp_path: Path) -> None:
    src = write_src(tmp_path / "src")
    manifest = data.build(src, tmp_path / "out", TEMPLATE)
    row = data.read_jsonl(tmp_path / "out" / "test.jsonl")[0]
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["messages"][1]["content"] == "Sort a lis"  # cut at 10 chars, start kept
    assert row["messages"][2]["content"] == row["label"] == "M"
    assert manifest["prompts_cut"] == 2  # "Write a haiku", "Sort a list"; "Who won X?" is 10
    assert manifest["prompt_template_sha256"] == TEMPLATE.sha256


def test_build_is_byte_identical(tmp_path: Path) -> None:
    src = write_src(tmp_path / "src")
    first = data.build(src, tmp_path / "a", TEMPLATE)
    second = data.build(src, tmp_path / "b", TEMPLATE)
    assert first["files"] == second["files"]
    assert b"\r\n" not in (tmp_path / "a" / "test.jsonl").read_bytes()


def test_sha_mismatch_is_refused(tmp_path: Path) -> None:
    src = write_src(tmp_path / "src", sha="0" * 64)
    with pytest.raises(data.DataError, match="does not match"):
        data.build(src, tmp_path / "out", TEMPLATE)
    assert not (tmp_path / "out").exists()  # nothing written


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"split": {"train": ["dolly-1", "mbpp-1"], "test": ["mbpp-1"]}}, "both train and test"),
        ({"labels": {k: v for k, v in LABELS.items() if k != "gsm8k-1"}}, "no label"),
        ({"split": {"train": ["dolly-1"], "test": ["mbpp-1"]}}, "neither split"),
        ({"prompts": [*PROMPTS, PROMPTS[0]]}, "duplicate prompt id"),
    ],
)
def test_inconsistent_input_is_refused(tmp_path: Path, kwargs: dict, message: str) -> None:
    src = write_src(tmp_path / "src", **kwargs)
    with pytest.raises(data.DataError, match=message):
        data.build(src, tmp_path / "out", TEMPLATE)


def test_template_hash_ignores_yaml_comments_but_not_wording(tmp_path: Path) -> None:
    base = (ROOT / "prompts" / "router_v1.yaml").read_text(encoding="utf-8")
    original = load_template(ROOT / "prompts" / "router_v1.yaml")
    folder = tmp_path / "prompts"
    folder.mkdir()
    path = folder / "router_v1.yaml"

    path.write_text("# a new comment\n" + base, encoding="utf-8")
    assert load_template(path).sha256 == original.sha256

    path.write_text(base.replace("one letter", "a single letter"), encoding="utf-8")
    assert load_template(path).sha256 != original.sha256


def test_template_file_name_must_match(tmp_path: Path) -> None:
    folder = tmp_path / "prompts"
    folder.mkdir()
    text = (ROOT / "prompts" / "router_v1.yaml").read_text(encoding="utf-8")
    (folder / "router_v2.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="must be named router_v1.yaml"):
        load_registry(folder)


def test_repo_registry_has_router_v1() -> None:
    template = get_template("router:1", ROOT / "prompts")
    assert "S, M or L" in template.system


def test_cli_build_and_show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)  # the default --prompt router:1 is read from ./prompts
    runner = CliRunner()
    src = write_src(tmp_path / "src")
    out = tmp_path / "out"
    built = runner.invoke(app, ["data", "build", "--src", str(src), "--out", str(out)])
    assert built.exit_code == 0, built.output
    shown = runner.invoke(app, ["data", "show", "--out", str(out)])
    assert shown.exit_code == 0
    assert "train_old" in shown.output


def test_cli_show_without_data(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["data", "show", "--out", str(tmp_path)])
    assert result.exit_code == 1

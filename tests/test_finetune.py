from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pitwall.cli import app
from pitwall.data import Pair
from pitwall.finetune import bundle, core, llamacpp
from pitwall.textformat import ANSWERS, FORMAT_VERSION, render, render_pair

ROOT = Path(__file__).resolve().parents[1]


def pair(i: int, label: str, prompt: str = "hello") -> Pair:
    return Pair(
        id=f"p{i:03d}",
        messages=[
            {"role": "system", "content": "Pick a tier."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label},
        ],
        label=label,  # type: ignore[arg-type]
        task_type="writing",
        source="dolly",
        part="old",
    )


# --- text format ---------------------------------------------------------------------


def test_render_layout_is_fixed() -> None:
    assert render("SYS", "what?") == "SYS\n\nRequest:\nwhat?\n\nTier:"


def test_render_pair_gives_prompt_and_spaced_answer() -> None:
    text, answer = render_pair(pair(1, "L", "Prove it."))
    assert text.endswith("Request:\nProve it.\n\nTier:")
    assert answer == " L" == ANSWERS["L"]


# --- validation split ------------------------------------------------------------------


def test_split_is_stratified_and_repeatable() -> None:
    pairs = [pair(i, label) for i, label in enumerate("S" * 50 + "M" * 40 + "L" * 3)]
    train, val = core.validation_split(pairs, 0.2, seed=1)
    again = core.validation_split(pairs, 0.2, seed=1)
    assert [p.id for p in val] == [p.id for p in again[1]]
    assert {p.id for p in train}.isdisjoint({p.id for p in val})
    assert len(train) + len(val) == len(pairs)
    counts = {t: sum(p.label == t for p in val) for t in "SML"}
    assert counts == {"S": 10, "M": 8, "L": 1}  # rare L still in validation


def test_split_changes_with_seed() -> None:
    pairs = [pair(i, "S") for i in range(40)]
    a = core.validation_split(pairs, 0.2, seed=1)[1]
    b = core.validation_split(pairs, 0.2, seed=2)[1]
    assert [p.id for p in a] != [p.id for p in b]


# --- loss mask -------------------------------------------------------------------------


def test_example_is_prompt_and_one_answer_token() -> None:
    assert core.make_example([5, 6, 7], [99], max_len=10) == ([5, 6, 7], 99)


def test_answer_must_be_one_token() -> None:
    with pytest.raises(ValueError, match="must be one"):
        core.make_example([5], [1, 2], max_len=10)


def test_too_long_example_is_an_error() -> None:
    with pytest.raises(ValueError, match="over max_len"):
        core.make_example(list(range(10)), [1], max_len=10)


# --- choosing epochs and threshold -----------------------------------------------------------


def test_choose_prefers_lowest_cost_then_fewer_epochs() -> None:
    labels = np.array([0, 1, 2])
    perfect = np.eye(3) * 0.9 + 0.05
    poor = np.tile([0.1, 0.8, 0.1], (3, 1))
    results = [
        core.EpochResult(1, poor, labels),
        core.EpochResult(2, perfect, labels),
        core.EpochResult(3, perfect, labels),
    ]
    best, rows = core.choose(results)
    assert best["epoch"] == 2  # epoch 3 is as good; fewer epochs win the tie
    assert best["cost"] == 0.0
    assert len(rows) == 3 * 4  # epochs x threshold grid


def test_renormalise_is_a_softmax() -> None:
    p = core.renormalise(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(p[0], [1 / 3] * 3)
    assert p[1, 0] > 0.99


# --- bundle ----------------------------------------------------------------------------


def make_bundle(tmp_path: Path, gguf_bytes: bytes = b"GGUF fake model") -> Path:
    directory = tmp_path / "ft-test"
    (directory / bundle.ADAPTER_DIR).mkdir(parents=True)
    (directory / bundle.MODEL_FILE).write_bytes(gguf_bytes)
    manifest = bundle.Manifest(
        name="ft-test",
        created_at="2026-11-06T00:00:00+00:00",
        git_commit="abc",
        git_dirty=False,
        base_model="Qwen/Qwen3-0.6B-Base",
        base_revision="da87bfb",
        text_format=FORMAT_VERSION,
        answer_token_ids={"S": 1, "M": 2, "L": 3},
        prompt_template="router:1",
        prompt_sha256="x" * 64,
        data_version="709cd348.dir",
        train_file="data/derived/train_old.jsonl",
        train_sha256="y" * 64,
        examples=365,
        validation_examples=73,
        lora={"r": 16},
        training={"lr": 2e-4},
        chosen={
            "epoch": 2,
            "threshold": 0.5,
            "cost": 0.5,
            "accuracy": 0.5,
            "under_rate": 0.1,
            "picks": {"S": 1, "M": 70, "L": 2},
        },
        validation=[],
        libraries={"torch": "2.x"},
        gguf=bundle.GgufInfo(
            quant="Q4_K_M",
            sha256=bundle.sha256_file(directory / bundle.MODEL_FILE),
            bytes=len(gguf_bytes),
            llama_cpp_tag=llamacpp.TAG,
            llama_cpp_commit=llamacpp.COMMIT,
        ),
    )
    bundle.write_manifest(directory, manifest)
    return directory


def test_good_bundle_passes(tmp_path: Path) -> None:
    directory = make_bundle(tmp_path)
    assert bundle.check(directory) == []
    result = CliRunner().invoke(app, ["bundle", "check", str(directory)])
    assert result.exit_code == 0, result.output
    assert "2 epochs, threshold 0.5" in result.output


def test_replaced_model_is_caught(tmp_path: Path) -> None:
    directory = make_bundle(tmp_path)
    (directory / bundle.MODEL_FILE).write_bytes(b"GGUF fake modeX")  # same size, new bytes
    problems = bundle.check(directory)
    assert any("sha256 does not match" in p for p in problems)
    assert CliRunner().invoke(app, ["bundle", "check", str(directory)]).exit_code == 1


def test_wrong_text_format_and_missing_parts(tmp_path: Path) -> None:
    directory = make_bundle(tmp_path)
    data = json.loads((directory / bundle.MANIFEST).read_text(encoding="utf-8"))
    data["text_format"] = "something-else/9"
    (directory / bundle.MANIFEST).write_text(json.dumps(data), encoding="utf-8")
    (directory / bundle.ADAPTER_DIR).rmdir()
    problems = bundle.check(directory)
    assert any("text format" in p for p in problems)
    assert any("adapter/ is missing" in p for p in problems)


def test_missing_manifest(tmp_path: Path) -> None:
    assert bundle.check(tmp_path) == [f"{tmp_path / bundle.MANIFEST} not found"]


# --- pins and notebook ---------------------------------------------------------------------


def test_llama_cpp_pins_look_right() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", llamacpp.COMMIT)
    for name, digest in llamacpp.ASSETS.values():
        assert llamacpp.TAG in name
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_notebook_runs_the_script_and_stops_on_failure() -> None:
    nb = json.loads((ROOT / "notebooks" / "train_router_colab.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    train_cells = [c for c in code if "pitwall.finetune.train" in c]
    assert len(train_cells) == 1
    # every shell step that matters is followed by a check (Run all doesn't stop otherwise)
    for cell in code:
        if any(k in cell for k in ("git checkout", "pip install", "dvc repro", "finetune.train")):
            assert "assert _exit_code == 0" in cell, cell


def test_colab_requirements_are_pinned_and_skip_torch() -> None:
    lines = [
        line.strip()
        for line in (ROOT / "training" / "requirements-colab.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert all("==" in line for line in lines)
    assert not any(line.startswith("torch") for line in lines)


def test_notebook_removes_colabs_old_torchao() -> None:
    # peft refuses to load beside torchao < 0.16, and Colab ships 0.10 (first Colab run).
    nb = json.loads((ROOT / "notebooks" / "train_router_colab.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert code.index("pip uninstall -q -y torchao") < code.index("pitwall.finetune.train")


def test_notebook_stops_without_a_gpu() -> None:
    nb = json.loads((ROOT / "notebooks" / "train_router_colab.ipynb").read_text(encoding="utf-8"))
    first = "".join(next(c["source"] for c in nb["cells"] if c["cell_type"] == "code"))
    assert "nvidia-smi" in first and "assert _exit_code == 0" in first

"""Build Pitwall's training and test files from Gearbox's labelled dataset.

Input (data/gearbox/, imported with DVC from a pinned Gearbox commit): prompts.jsonl,
labels.jsonl (cheapest tier that answered well), split.json, manifest.json.

Output (data/derived/), each row a chat-format instruction pair whose answer is the tier:
- test.jsonl         Gearbox's test split. The same for every model, always.
- train_old.jsonl    train rows from the general sources: what the first model learns.
- pool_new.jsonl     train rows from the hard sources (SimpleQA, AIME, MMLU-Pro): the
                     "new traffic" of the drift demo, already labelled by Gearbox.
- train_full.jsonl   old + new: what the retrained model learns.
- manifest.json      counts, sha256 of every file, the source and the prompt hash.

Deterministic (rows sorted by id, fixed JSON formatting), so `dvc repro` on any machine
gives the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from pitwall.prompts import PromptTemplate

Tier = Literal["S", "M", "L"]
Part = Literal["old", "hard"]

SRC_DIR = Path("data/gearbox")
OUT_DIR = Path("data/derived")
# The sources Gearbox added in dataset v2 to find prompts only the big model gets right.
HARD_SOURCES = frozenset({"simpleqa", "aime2025", "mmlu-pro"})
FILES = ("test", "train_old", "pool_new", "train_full")


class DataError(ValueError):
    """The input files are inconsistent; nothing is written."""


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    messages: list[dict[str, str]]
    label: Tier
    task_type: str
    source: str
    part: Part


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    # newline="\n": the same bytes on Windows and Linux.
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(), ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def make_pair(prompt: dict[str, Any], label: Tier, template: PromptTemplate) -> Pair:
    return Pair(
        id=prompt["id"],
        messages=[
            {"role": "system", "content": template.system},
            {"role": "user", "content": template.render_user(prompt["prompt"])},
            {"role": "assistant", "content": label},
        ],
        label=label,
        task_type=prompt["task_type"],
        source=prompt["source"],
        part="hard" if prompt["source"] in HARD_SOURCES else "old",
    )


def _source_url(src: Path) -> str:
    # Where DVC imported prompts.jsonl from (the pinned Gearbox URL), for the manifest.
    dvc_file = src / "prompts.jsonl.dvc"
    if not dvc_file.exists():
        return "unknown"
    deps = (yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}).get("deps") or []
    return str(deps[0]["path"]) if deps else "unknown"


def load_and_check(src: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Tier], dict]:
    """Read the Gearbox files and refuse inconsistent input."""
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    actual = sha256_file(src / "prompts.jsonl")
    if actual != manifest["prompts_sha256"]:
        raise DataError(
            f"prompts.jsonl sha256 {actual[:12]} does not match Gearbox's manifest "
            f"{manifest['prompts_sha256'][:12]}"
        )

    prompts: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(src / "prompts.jsonl"):
        if row["id"] in prompts:
            raise DataError(f"duplicate prompt id {row['id']}")
        prompts[row["id"]] = row
    labels: dict[str, Tier] = {row["id"]: row["label"] for row in read_jsonl(src / "labels.jsonl")}
    split = json.loads((src / "split.json").read_text(encoding="utf-8"))
    train, test = set(split["train"]), set(split["test"])

    if overlap := train & test:
        raise DataError(f"{len(overlap)} ids are in both train and test, e.g. {min(overlap)}")
    if missing := (train | test) - prompts.keys():
        raise DataError(f"{len(missing)} split ids have no prompt, e.g. {min(missing)}")
    if unlabelled := (train | test) - labels.keys():
        raise DataError(f"{len(unlabelled)} split ids have no label, e.g. {min(unlabelled)}")
    if unsplit := prompts.keys() - train - test:
        raise DataError(f"{len(unsplit)} prompts are in neither split, e.g. {min(unsplit)}")
    return prompts, labels, split


def build(src: Path, out: Path, template: PromptTemplate) -> dict[str, Any]:
    prompts, labels, split = load_and_check(src)

    def pairs(ids: list[str]) -> list[Pair]:
        return [make_pair(prompts[i], labels[i], template) for i in sorted(ids)]

    test = pairs(split["test"])
    train = pairs(split["train"])
    files: dict[str, list[Pair]] = {
        "test": test,
        "train_old": [p for p in train if p.part == "old"],
        "pool_new": [p for p in train if p.part == "hard"],
        "train_full": train,
    }

    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name, rows in files.items():
        path = out / f"{name}.jsonl"
        write_jsonl(path, rows)
        summary[name] = {
            "rows": len(rows),
            "sha256": sha256_file(path),
            "by_label": dict(sorted(Counter(p.label for p in rows).items())),
            "by_part": dict(sorted(Counter(p.part for p in rows).items())),
            "by_task_type": dict(sorted(Counter(p.task_type for p in rows).items())),
        }

    cut = sum(len(prompts[i]["prompt"]) > template.max_prompt_chars for i in prompts)
    manifest = {
        "source": _source_url(src),
        "gearbox_prompts_sha256": sha256_file(src / "prompts.jsonl"),
        "prompt_template": template.key,
        "prompt_template_sha256": template.sha256,
        "max_prompt_chars": template.max_prompt_chars,
        "prompts_cut": cut,
        "hard_sources": sorted(HARD_SOURCES),
        "files": summary,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def table(manifest: dict[str, Any]) -> str:
    """Plain-ASCII summary for `pitwall data show` (cp1252 consoles)."""
    lines = [
        f"source:  {manifest['source']}",
        f"prompt:  {manifest['prompt_template']} ({manifest['prompt_template_sha256'][:12]}), "
        f"{manifest['prompts_cut']} prompts cut at {manifest['max_prompt_chars']} chars",
        "",
        f"{'file':<12} {'rows':>5}  {'S':>4} {'M':>4} {'L':>4}  {'old':>4} {'hard':>4}  sha256",
    ]
    for name in FILES:
        f = manifest["files"][name]
        lab, part = f["by_label"], f["by_part"]
        lines.append(
            f"{name:<12} {f['rows']:>5}  {lab.get('S', 0):>4} {lab.get('M', 0):>4} "
            f"{lab.get('L', 0):>4}  {part.get('old', 0):>4} {part.get('hard', 0):>4}  "
            f"{f['sha256'][:12]}"
        )
    return "\n".join(lines)

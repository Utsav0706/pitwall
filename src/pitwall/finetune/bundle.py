"""The bundle a training run produces, and the check every later step runs first.

    bundles/<name>/
      model.gguf     the merged model, quantised (Q4_K_M): what Ollama serves
      adapter/       the LoRA weights alone (a few MB), to inspect or re-merge
      bundle.json    everything needed to say how this model was made

Training happens on Colab; evaluation, registration and serving happen locally. The
bundle is the hand-over between the two, so it carries its own proof: the GGUF's
sha256 and size, and the data, prompt and code versions it was trained from.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pitwall.textformat import FORMAT_VERSION

BUNDLE_FORMAT = "pitwall-bundle/1"
MANIFEST = "bundle.json"
MODEL_FILE = "model.gguf"
ADAPTER_DIR = "adapter"


class GgufInfo(BaseModel):
    file: str = MODEL_FILE
    quant: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    llama_cpp_tag: str
    llama_cpp_commit: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str = BUNDLE_FORMAT
    name: str
    created_at: str
    git_commit: str
    git_dirty: bool
    base_model: str
    base_revision: str
    text_format: str
    answer_token_ids: dict[str, int]
    prompt_template: str
    prompt_sha256: str
    data_version: str
    train_file: str
    train_sha256: str
    examples: int
    validation_examples: int
    lora: dict[str, Any]
    training: dict[str, Any]
    chosen: dict[str, Any]  # epochs and threshold picked on validation, with its metrics
    validation: list[dict[str, Any]]  # every (epoch, threshold) row
    libraries: dict[str, str]
    gguf: GgufInfo | None  # None only for a smoke run without llama.cpp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(directory: Path, manifest: Manifest) -> Path:
    path = directory / MANIFEST
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def load_manifest(directory: Path) -> Manifest:
    return Manifest.model_validate_json((directory / MANIFEST).read_text(encoding="utf-8"))


def check(directory: Path, require_gguf: bool = True) -> list[str]:
    """Problems with a bundle; an empty list means it is complete and intact."""
    if not (directory / MANIFEST).exists():
        return [f"{directory / MANIFEST} not found"]
    try:
        manifest = load_manifest(directory)
    except (ValidationError, json.JSONDecodeError) as exc:
        return [f"{MANIFEST} is invalid: {exc}"]
    problems = []
    if manifest.format != BUNDLE_FORMAT:
        problems.append(f"format {manifest.format!r}, expected {BUNDLE_FORMAT!r}")
    if manifest.text_format != FORMAT_VERSION:
        problems.append(
            f"trained with text format {manifest.text_format!r}; this code serves "
            f"{FORMAT_VERSION!r}"
        )
    if set(manifest.answer_token_ids) != {"S", "M", "L"}:
        problems.append("answer_token_ids must have exactly S, M and L")
    if not (directory / ADAPTER_DIR).is_dir():
        problems.append(f"{ADAPTER_DIR}/ is missing")
    if manifest.gguf is None:
        if require_gguf:
            problems.append("no GGUF in this bundle (smoke run?)")
        return problems
    gguf = directory / manifest.gguf.file
    if not gguf.exists():
        problems.append(f"{gguf.name} is missing")
    elif gguf.stat().st_size != manifest.gguf.bytes:
        problems.append(
            f"{gguf.name} is {gguf.stat().st_size} bytes, manifest says {manifest.gguf.bytes}"
        )
    elif sha256_file(gguf) != manifest.gguf.sha256:
        problems.append(f"{gguf.name} sha256 does not match the manifest (corrupt or replaced)")
    return problems

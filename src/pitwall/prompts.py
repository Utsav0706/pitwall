"""A small prompt registry: one YAML file per template version, identified by a hash.

Git already keeps each file's history. The hash adds proof of exactly which wording a
dataset or model run used: it is written into the data manifest and (from M2) into
every MLflow run. A file must be named <name>_v<version>.yaml, so there is exactly one
"router v1"; a new wording is a new version, never an edit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROMPTS_DIR = Path("prompts")


class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    version: int = Field(ge=1)
    system: str
    user_template: str
    max_prompt_chars: int = Field(gt=0)

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def sha256(self) -> str:
        # Canonical JSON of the fields, not the file bytes: comments and key order in
        # the YAML don't change what the model sees, so they don't change the hash.
        canonical = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def render_user(self, prompt: str) -> str:
        return self.user_template.format(prompt=prompt[: self.max_prompt_chars])


def load_template(path: Path) -> PromptTemplate:
    template = PromptTemplate.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    expected = f"{template.name}_v{template.version}.yaml"
    if path.name != expected:
        raise ValueError(f"{path}: file for {template.key} must be named {expected}")
    return template


def load_registry(directory: Path = PROMPTS_DIR) -> dict[str, PromptTemplate]:
    registry: dict[str, PromptTemplate] = {}
    for path in sorted(directory.glob("*.yaml")):
        template = load_template(path)
        registry[template.key] = template
    return registry


def get_template(key: str, directory: Path = PROMPTS_DIR) -> PromptTemplate:
    registry = load_registry(directory)
    if key not in registry:
        raise KeyError(f"unknown prompt {key!r}; available: {sorted(registry)}")
    return registry[key]

"""Two kinds of settings, kept apart on purpose.

- `Settings`: what differs between machines (URLs, ports), from the environment or a
  local .env file. Secrets would live here too; Pitwall needs none yet.
- `PitwallConfig`: the rules (latency limit, canary share, drift threshold), from
  config.yaml in git, so a change to a rule is a reviewed commit, not a local edit.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: a blank line such as "GEARBOX_URL=" means "not set".
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    # Local MLflow server (M2): `mlflow server` with a SQLite backend, no Docker.
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    # Ollama serves the quantised model on CPU (M4).
    ollama_base_url: str = "http://127.0.0.1:11434"
    # Gearbox calls Pitwall's /route; Pitwall calls Gearbox for canary checks (M7).
    gearbox_url: str = "http://127.0.0.1:8002"
    # Dyno's run store, for the promotion gate (M6).
    dyno_database_url: str = "sqlite:///dyno.db"
    # Windows reserves 8000; Scout uses 8001, Gearbox 8002.
    pitwall_port: int = 8003
    log_level: str = "WARNING"


@lru_cache
def get_settings() -> Settings:
    return Settings()


class _Strict(BaseModel):
    # A typo in config.yaml is an error, not a silently ignored key.
    model_config = ConfigDict(extra="forbid")


class PromotionRules(_Strict):
    # A candidate reaches production only if it is no slower than this at p95 (M6).
    p95_ms_limit: float = Field(gt=0)
    # Largest allowed accuracy drop vs production, in percentage points (Dyno gate).
    max_drop_pp: float = Field(ge=0)


class CanaryRules(_Strict):
    # Share of Gearbox traffic sent to the candidate model (M7).
    percent: float = Field(gt=0, le=100)


class DriftRules(_Strict):
    # Share of drifted features at which an alert fires (M9).
    threshold: float = Field(gt=0, le=1)


class PitwallConfig(_Strict):
    promotion: PromotionRules
    canary: CanaryRules
    drift: DriftRules


DEFAULT_CONFIG = Path("config.yaml")


def load_config(path: Path = DEFAULT_CONFIG) -> PitwallConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PitwallConfig.model_validate(data)

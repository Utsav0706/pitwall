"""The route log (one row per routing decision) and drift check results.

Kept because later steps need it: drift detection (M9) embeds the prompts, and failing
requests become new training data (M10). The prompt text is stored as the model saw it
(cut to the template's max_prompt_chars) in a local database; it is never sent anywhere.

`open_engine` applies Alembic migrations every time, so a fresh checkout or an old
pitwall.db is brought up to date with no separate step (the same approach as Dyno).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Boolean, DateTime, Double, Engine, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


class Base(DeclarativeBase):
    pass


class RouteLog(Base):
    __tablename__ = "route_log"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_ref: Mapped[str] = mapped_column(String(32))  # what was asked: production, v1, ...
    model_version: Mapped[int] = mapped_column(Integer, index=True)  # what answered
    tier: Mapped[str] = mapped_column(String(1))
    p_s: Mapped[float] = mapped_column(Double)
    p_m: Mapped[float] = mapped_column(Double)
    p_l: Mapped[float] = mapped_column(Double)
    fallback: Mapped[bool] = mapped_column(Boolean)  # no S/M/L from the model -> M
    latency_ms: Mapped[float] = mapped_column(Double)
    prompt_chars: Mapped[int] = mapped_column(Integer)  # before the cut
    prompt_cut: Mapped[bool] = mapped_column(Boolean)
    prompt: Mapped[str] = mapped_column(Text)  # as the model saw it


class DriftCheck(Base):
    """One `pitwall drift check` result (M9); the service exposes the latest as gauges."""

    __tablename__ = "drift_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    model_ref: Mapped[str] = mapped_column(String(32))
    reference: Mapped[str] = mapped_column(String(200))  # bundle + its training file
    window_n: Mapped[int] = mapped_column(Integer)
    share: Mapped[float] = mapped_column(Double)  # share of drifted features
    detected: Mapped[bool] = mapped_column(Boolean)
    details: Mapped[str] = mapped_column(Text)  # JSON: per-feature values, task mix


def alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def open_engine(url: str) -> Engine:
    engine = create_engine(url)
    cfg = alembic_config(url)
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
    return engine

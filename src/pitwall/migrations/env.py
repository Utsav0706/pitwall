"""Alembic environment.

Pitwall runs migrations itself (`pitwall.serving.db.open_engine` passes its open
connection in `config.attributes`). The `alembic` command (alembic.ini at the repo root,
for writing new revisions) falls back to DATABASE_URL from the settings.
No logging.config.fileConfig here: it would replace Pitwall's JSON logging.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from pitwall.serving.db import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from pitwall.config import get_settings

    return get_settings().database_url


def _run(connection) -> None:  # type: ignore[no-untyped-def]
    # Batch mode lets ALTER-style changes work on SQLite (it copies the table).
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
elif (conn := config.attributes.get("connection")) is not None:
    _run(conn)
else:
    with create_engine(_url()).connect() as conn:
        _run(conn)
        conn.commit()

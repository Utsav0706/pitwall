"""route log

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "route_log",
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_ref", sa.String(length=32), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("tier", sa.String(length=1), nullable=False),
        sa.Column("p_s", sa.Double(), nullable=False),
        sa.Column("p_m", sa.Double(), nullable=False),
        sa.Column("p_l", sa.Double(), nullable=False),
        sa.Column("fallback", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Double(), nullable=False),
        sa.Column("prompt_chars", sa.Integer(), nullable=False),
        sa.Column("prompt_cut", sa.Boolean(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("request_id"),
    )
    op.create_index("ix_route_log_created_at", "route_log", ["created_at"])
    op.create_index("ix_route_log_model_version", "route_log", ["model_version"])


def downgrade() -> None:
    op.drop_index("ix_route_log_model_version", table_name="route_log")
    op.drop_index("ix_route_log_created_at", table_name="route_log")
    op.drop_table("route_log")

"""drift checks

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drift_checks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_ref", sa.String(length=32), nullable=False),
        sa.Column("reference", sa.String(length=200), nullable=False),
        sa.Column("window_n", sa.Integer(), nullable=False),
        sa.Column("share", sa.Double(), nullable=False),
        sa.Column("detected", sa.Boolean(), nullable=False),
        sa.Column("details", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drift_checks_created_at", "drift_checks", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_drift_checks_created_at", table_name="drift_checks")
    op.drop_table("drift_checks")

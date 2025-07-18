"""add movies column to user_profiles

Revision ID: 7110b2bfcb63
Revises: 3fc134fcd13f
Create Date: 2025-07-13 11:37:03.976877

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7110b2bfcb63"
down_revision: Union[str, Sequence[str], None] = "3fc134fcd13f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("user_profiles", sa.Column("movies", sa.Text(), nullable=True))
    op.alter_column(
        "user_profiles",
        "clusters",
        existing_type=postgresql.JSON(astext_type=sa.Text()),
        type_=sa.Text(),
        nullable=True,
    )
    op.drop_index(op.f("ix_user_profiles_user_id"), table_name="user_profiles")
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index(
        op.f("ix_user_profiles_user_id"), "user_profiles", ["user_id"], unique=False
    )
    op.alter_column(
        "user_profiles",
        "clusters",
        existing_type=sa.Text(),
        type_=postgresql.JSON(astext_type=sa.Text()),
        nullable=False,
    )
    op.drop_column("user_profiles", "movies")
    # ### end Alembic commands ###

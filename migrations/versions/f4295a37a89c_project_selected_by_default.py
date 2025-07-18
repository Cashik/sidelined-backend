"""project_selected_by_default

Revision ID: f4295a37a89c
Revises: 1d77903a1582
Create Date: 2025-06-17 13:08:44.207829

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f4295a37a89c'
down_revision: Union[str, None] = '1d77903a1582'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('project', sa.Column('is_selected_by_default', sa.Boolean(), server_default='false', nullable=False))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('project', 'is_selected_by_default')
    # ### end Alembic commands ###

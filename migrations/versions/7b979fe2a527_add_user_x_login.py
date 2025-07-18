"""add_user_x_login

Revision ID: 7b979fe2a527
Revises: 2c77f576da2c
Create Date: 2025-06-10 03:37:05.411042

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7b979fe2a527'
down_revision: Union[str, None] = '2c77f576da2c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('user', sa.Column('twitter_login', sa.String(), nullable=True))
    op.create_unique_constraint(None, 'user', ['twitter_login'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'user', type_='unique')
    op.drop_column('user', 'twitter_login')
    # ### end Alembic commands ###

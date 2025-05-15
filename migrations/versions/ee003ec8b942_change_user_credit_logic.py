"""change_user_credit_logic

Revision ID: ee003ec8b942
Revises: 6893fd8d0650
Create Date: 2025-05-14 18:38:16.537869

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ee003ec8b942'
down_revision: Union[str, None] = '6893fd8d0650'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Добавляем столбец с nullable=True и default=0
    op.add_column('user', sa.Column('used_credits_today', sa.Integer(), nullable=True, server_default='0'))
    # 2. Обновляем все строки, устанавливая used_credits_today = 0 (на случай, если default не сработает для существующих)
    op.execute('UPDATE "user" SET used_credits_today = 0')
    # 3. Делаем столбец NOT NULL
    op.alter_column('user', 'used_credits_today', nullable=False)
    # 4. Удаляем столбец credits
    op.drop_column('user', 'credits')


def downgrade() -> None:
    raise Exception("Downgrade is not supported")

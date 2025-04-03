"""addmodel

Revision ID: 4d9c21ec2901
Revises: 4514d0759889
Create Date: 2025-04-02 21:42:55.702532

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from alembic_postgresql_enum import TableReference

# revision identifiers, used by Alembic.
revision: str = '4d9c21ec2901'
down_revision: Union[str, None] = '4514d0759889'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values(
        enum_schema='public',
        enum_name='model',
        new_values=['GPT_4', 'GPT_4O', 'GPT_4O_MINI', 'GEMINI_2_FLASH'],
        affected_columns=[TableReference(table_schema='public', table_name='message', column_name='model'), TableReference(table_schema='public', table_name='user', column_name='preferred_chat_model')],
        enum_values_to_rename=[],
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.sync_enum_values(
        enum_schema='public',
        enum_name='model',
        new_values=['GPT_4', 'GPT_4O', 'GPT_4O_MINI'],
        affected_columns=[TableReference(table_schema='public', table_name='message', column_name='model'), TableReference(table_schema='public', table_name='user', column_name='preferred_chat_model')],
        enum_values_to_rename=[],
    )
    # ### end Alembic commands ###

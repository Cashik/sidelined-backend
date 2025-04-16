"""multiaddress

Revision ID: 73f2f6b291d5
Revises: a508effc3d12
Create Date: 2025-04-15 18:27:35.559854

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = '73f2f6b291d5'
down_revision: Union[str, None] = 'a508effc3d12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Проверяем наличие дубликатов адресов
    conn = op.get_bind()
    duplicates = conn.execute(text("""
        WITH duplicate_addresses AS (
            SELECT LOWER(address) as lower_address, COUNT(*) as count
            FROM "user"
            WHERE address IS NOT NULL
            GROUP BY LOWER(address)
            HAVING COUNT(*) > 1
        )
        SELECT u.id, LOWER(u.address) as address, 
               (SELECT COUNT(*) FROM chat WHERE chat.user_id = u.id) as chat_count
        FROM "user" u
        JOIN duplicate_addresses da ON LOWER(u.address) = da.lower_address
        ORDER BY LOWER(u.address), chat_count DESC
    """)).fetchall()
    
    if duplicates:
        print("\nНайдены дубликаты адресов:")
        current_address = None
        users_to_delete = []
        
        for dup in duplicates:
            if current_address != dup[1]:
                print(f"\nАдрес: {dup[1]}")
                current_address = dup[1]
                # Первый пользователь с этим адресом (с наибольшим количеством чатов) будет сохранен
                print(f"  Сохраняем пользователя {dup[0]} с {dup[2]} чатами")
            else:
                # Остальных пользователей с этим адресом добавляем в список на удаление
                print(f"  Удаляем пользователя {dup[0]} с {dup[2]} чатами")
                users_to_delete.append(dup[0])
        
        if users_to_delete:
            # Удаляем дубликаты
            print(f"\nУдаляем {len(users_to_delete)} пользователей-дубликатов...")
            conn.execute(text("""
                DELETE FROM "user" 
                WHERE id = ANY(:user_ids)
            """), {"user_ids": users_to_delete})
            print("Удаление завершено")
    
    # Создаем новую таблицу для адресов
    op.create_table('wallet_address',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('address', sa.String(), nullable=True),
        sa.Column('created_at', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('address')
    )
    op.create_index(op.f('ix_wallet_address_id'), 'wallet_address', ['id'], unique=False)
    
    # Переносим существующие адреса в новую таблицу
    conn.execute(text("""
        INSERT INTO wallet_address (user_id, address, created_at)
        SELECT id, LOWER(address), created_at FROM "user"
        WHERE address IS NOT NULL
    """))
    
    # Удаляем колонку address из таблицы user
    op.drop_column('user', 'address')


def downgrade() -> None:
    # Проверяем, есть ли адреса в новой таблице
    conn = op.get_bind()
    result = conn.execute(text("SELECT COUNT(*) FROM wallet_address")).scalar()
    
    if result > 0:
        raise Exception("Cannot downgrade: There are addresses in the wallet_address table. This would cause data loss.")
    
    # Если адресов нет, выполняем даунгрейд
    op.add_column('user', sa.Column('address', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.drop_index(op.f('ix_wallet_address_id'), table_name='wallet_address')
    op.drop_table('wallet_address')

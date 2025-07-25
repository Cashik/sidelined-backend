"""add_Aethir

Revision ID: 209eda1d310b
Revises: d392e36ecd93
Create Date: 2025-07-25 11:05:58.969411

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import time


# revision identifiers, used by Alembic.
revision: str = '209eda1d310b'
down_revision: Union[str, None] = 'd392e36ecd93'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    
    # Получаем текущее время как timestamp
    now_timestamp = int(time.time())
    
    # 1. Добавляем проект Aethir (если его нет)
    op.execute(f"""
        INSERT INTO project (created_at, name, description, url, icon_url, keywords, search_min_likes, is_selected_by_default, is_leaderboard_project)
        SELECT {now_timestamp}, 'Aethir', NULL, 'https://aethir.com/', 'https://assets.coingecko.com/coins/images/36179/standard/logogram_circle_dark_green_vb_green_%281%29.png?1718232706', 'Aethir;$ATH;@AethirCloud', 0, FALSE, TRUE
        WHERE NOT EXISTS (
            SELECT 1 FROM project WHERE name = 'Aethir'
        );
    """)
    
    # 2. Добавляем социальный аккаунт @AethirCloud (если его нет)
    op.execute(f"""
        INSERT INTO social_account (created_at, social_id, social_login, name, twitter_scout_score, twitter_scout_score_updated_at, last_avatar_url, last_followers_count, is_disabled_for_leaderboard)
        SELECT {now_timestamp}, '', 'AethirCloud', 'Aethir', NULL, NULL, NULL, NULL, FALSE
        WHERE NOT EXISTS (
            SELECT 1 FROM social_account WHERE social_login = 'AethirCloud'
        );
    """)
    
    # 3. Создаем связь между проектом и аккаунтом как MEDIA (если ее нет)
    op.execute(f"""
        INSERT INTO project_account_status (created_at, project_id, account_id, type)
        SELECT {now_timestamp}, p.id, s.id, 'MEDIA'
        FROM project p, social_account s
        WHERE p.name = 'Aethir' AND s.social_login = 'AethirCloud'
        AND NOT EXISTS (
            SELECT 1 FROM project_account_status pas
            WHERE pas.project_id = p.id AND pas.account_id = s.id
        );
    """)
    
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Даунгрейд не должен ничего удалять по требованию
    pass
    # ### end Alembic commands ###

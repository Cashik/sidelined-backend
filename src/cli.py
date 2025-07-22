import asyncio
import argparse
import logging
import bcrypt
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session, aliased
import requests
import json
from collections import defaultdict

from src.database import SessionLocal
from src.models import User, Project, AdminUser
from src import crud, utils_base, enums, models, utils
from src.config.projects import projects_all
from src.exceptions import PostTextExtractionError

from src.config.settings import settings

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def delete_all_users(force: bool = False):
    """
    Удалить всех пользователей из базы данных
    """
    if not force:
        confirm = input("Вы уверены, что хотите удалить всех пользователей? Это действие нельзя отменить. (y/n): ")
        if confirm.lower() != 'y':
            print("Операция отменена")
            return

    session = SessionLocal()
    try:
        stmt = select(User)
        users = session.execute(stmt).scalars().all()
        deleted_count = 0
        for user in users:
            try:
                asyncio.run(crud.delete_user(user, session))
                deleted_count += 1
            except Exception as e:
                print(f"Ошибка при удалении пользователя {user.id}: {str(e)}")
                continue
        
        print(f"Успешно удалено {deleted_count} пользователей")
    finally:
        session.close()


def create_promo_code(code: str, valid_until: int):
    session = SessionLocal()
    code = utils_base.format_promo_code(code)
    try:
        crud.create_promo_code(session, code, valid_until)
        print(f"Промо-код {code} успешно создан")
    finally:
        session.close()
        

def change_promo_code(code: str, valid_until: int):
    session = SessionLocal()
    code = utils_base.format_promo_code(code)
    try:
        crud.change_promo_code(session, code, valid_until)
        print(f"Промо-код {code} успешно изменен")
    finally:
        session.close()


def sync_projects():
    """
    Синхронизирует проекты из config/projects.py в базу данных.
    Создает или обновляет проекты, их социальные аккаунты и связи между ними.
    """
    session = SessionLocal()
    try:
        logger.info("Начало синхронизации проектов")
        for project in projects_all:
            try:
                logger.info(f"Обработка проекта: {project.name}")
                
                # создаем или обновляем проект в базе данных
                project_db: models.Project = models.Project(
                    name=project.name,
                    description=project.description,
                    url=project.url,
                    keywords=(";".join(project.keywords) if isinstance(project.keywords, (list, tuple)) else project.keywords),
                )
                project_db = crud.create_or_update_project_by_name(session, project_db)
                logger.info(f"Проект {project.name} успешно синхронизирован")
                
                # создаем или обновляем аккаунты проектов в базе данных
                for social_media in project.social_media:
                    try:
                        social_media_db: models.SocialAccount = models.SocialAccount(
                            name=social_media.name,
                            social_id=social_media.social_id,
                            social_login=social_media.social_login,
                        )
                        social_media_db = crud.create_or_update_social_media_by_social_name(session, social_media_db)
                        
                        # создаем или обновляем статус аккаунта в базе данных
                        crud.create_or_update_project_account_status(
                            session,
                            project_db.id,
                            social_media_db.id,
                            social_media.social_media_type
                        )
                        logger.info(f"Аккаунт {social_media.social_login} успешно синхронизирован")
                    except Exception as e:
                        logger.error(f"Ошибка при синхронизации аккаунта {social_media.social_login}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Ошибка при синхронизации проекта {project.name}: {str(e)}")
                continue
                
        logger.info("Синхронизация проектов завершена")
    except Exception as e:
        logger.error(f"Критическая ошибка при синхронизации проектов: {str(e)}")
        raise
    finally:
        session.close()


async def sync_posts_async():
    """
    Синхронизация постов из соц. сетей по всем проектам за последние 3 дня.
    """
    session = SessionLocal()
    try:
        # Получаем все проекты из базы данных
        projects = session.execute(select(models.Project)).scalars().all()
        from_timestamp = utils_base.now_timestamp() - settings.POST_SYNC_PERIOD_SECONDS
        # Для каждого проекта запускаем синхронизацию постов
        for project in projects:
            logger.info(f"sync_posts: project={project.name} start")
            feed_sync = await utils.update_project_feed(project, from_timestamp, session)
            news_sync = await utils.update_project_news(project, from_timestamp, session)
            logger.info(f"sync_posts: project={project.name} done => news_sync={news_sync}")
    finally:
        session.close()

def sync_posts():
    """
    Синхронизация постов из соц. сетей по всем проектам за последние 3 дня.
    """
    asyncio.run(sync_posts_async())


async def cleanup_old_posts_async(dry_run: bool = False):
    """
    Очистка старых постов из базы данных.
    """
    from src.config.settings import settings
    
    session = SessionLocal()
    try:
        cutoff_timestamp = utils_base.now_timestamp() - settings.POST_TO_TRASH_LIFETIME_SECONDS
        
        if dry_run:
            # В режиме dry-run только показываем, что будет удалено
            stmt = select(models.SocialPost).where(models.SocialPost.posted_at < cutoff_timestamp)
            posts_to_delete = session.execute(stmt).scalars().all()
            print(f"В режиме dry-run: найдено {len(posts_to_delete)} постов для удаления")
            print(f"Cutoff timestamp: {cutoff_timestamp} (посты старше {settings.POST_TO_TRASH_LIFETIME_SECONDS} секунд)")
            for post in posts_to_delete[:5]:  # Показываем первые 5 для примера
                print(f"  - Post ID: {post.id}, Posted at: {post.posted_at}, Text: {post.text[:50]}...")
            if len(posts_to_delete) > 5:
                print(f"  ... и ещё {len(posts_to_delete) - 5} постов")
        else:
            # Реальное удаление
            deleted_count = await crud.delete_old_posts(session, cutoff_timestamp)
            print(f"Успешно удалено {deleted_count} старых постов")
            print(f"Cutoff timestamp: {cutoff_timestamp} (посты старше {settings.POST_TO_TRASH_LIFETIME_SECONDS} секунд)")
    finally:
        session.close()


def cleanup_old_posts(dry_run: bool = False):
    """
    Очистка старых постов из базы данных.
    """
    asyncio.run(cleanup_old_posts_async(dry_run))


def create_admin_user(login: str, password: str, role: str):
    """Создать администратора или модератора"""
    role_upper = role.upper()
    if role_upper not in enums.AdminRole._value2member_map_:
        print(f"Недопустимая роль. Возможные значения: {', '.join([r.name.lower() for r in enums.AdminRole])}")
        return
    session = SessionLocal()
    try:
        # проверяем, что логин уникален
        existing = session.execute(select(AdminUser).where(AdminUser.login == login)).scalar_one_or_none()
        if existing:
            print(f"Пользователь с логином {login} уже существует")
            return
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        admin = AdminUser(
            login=login,
            password_hash=password_hash,
            role=enums.AdminRole(role_upper)
        )
        session.add(admin)
        session.commit()
        print(f"Администратор {login} успешно создан с ролью {role}")
    except Exception as e:
        session.rollback()
        print(f"Ошибка при создании администратора: {str(e)}")
    finally:
        session.close()


def delete_all_admin_users(force: bool = False):
    """Удалить всех администраторов (экстренная команда)"""
    if not force:
        confirm = input("Вы уверены, что хотите удалить ВСЕХ администраторов? Это действие нельзя отменить. (y/n): ")
        if confirm.lower() != 'y':
            print("Операция отменена")
            return
    session = SessionLocal()
    try:
        deleted = session.query(AdminUser).delete()
        session.commit()
        print(f"Удалено администраторов: {deleted}")
    except Exception as e:
        session.rollback()
        print(f"Ошибка при удалении администраторов: {str(e)}")
    finally:
        session.close()


def update_leaderboard(project_id: int = None):
    """
    Обновить лидерборд для одного или всех проектов с is_leaderboard_project=True
    """
    session = SessionLocal()
    try:
        if project_id is not None:
            project = session.query(Project).filter(Project.id == project_id).first()
            if not project:
                print(f"Проект с id={project_id} не найден")
                return
            print(f"Обновление лидерборда для проекта: {project.name} (id={project.id})")
            asyncio.run(utils.update_project_leaderboard(project, session))
        else:
            projects = session.query(Project).filter(Project.is_leaderboard_project == True).all()
            if not projects:
                print("Нет проектов с is_leaderboard_project=True")
                return
            for project in projects:
                print(f"Обновление лидерборда для проекта: {project.name} (id={project.id})")
                asyncio.run(utils.update_project_leaderboard(project, session))
    finally:
        session.close()


def update_users_xscore():
    """
    Обновить xscore для всех пользователей, чьи аккаунты упоминают лидербордские проекты и их xcore не установлен.
    """
    session = SessionLocal()
    try:
        print("Запуск обновления xscore пользователей...")
        utils.update_users_xscore_with_linked_accounts(session)
        print("Обновление xscore пользователей завершено.")
    finally:
        session.close()


def master_update():
    """
    Мастер-обновление: синк постов, обновление лидербордов, очистка старых постов.
    """
    session = SessionLocal()
    try:
        print("Запуск мастер-обновления...")
        asyncio.run(utils.master_update(session))
        print("Мастер-обновление завершено.")
    finally:
        session.close()


def create_auto_yaps(project_name: str = None):
    """
    Создать auto-yaps (авто-шаблоны постов) для проекта или для всех проектов.
    """
    session = SessionLocal()
    try:
        if project_name:
            project = session.query(Project).filter(Project.name == project_name).first()
            if not project:
                print(f"Проект с названием '{project_name}' не найден.")
                return
            print(f"Создание auto-yaps для проекта: {project.name}")
            result = asyncio.run(utils.create_project_autoyaps(project, session))
            print(f"Создано {len(result)} auto-yaps для проекта '{project.name}'")
        else:
            projects = session.query(Project).all()
            if not projects:
                print("В базе нет проектов.")
                return
            total = 0
            for project in projects:
                print(f"Создание auto-yaps для проекта: {project.name}")
                result = asyncio.run(utils.create_project_autoyaps(project, session))
                print(f"  - создано {len(result)} auto-yaps")
                total += len(result)
            print(f"Всего создано {total} auto-yaps для {len(projects)} проектов.")
    finally:
        session.close()


def update_social_accounts_profile():
    """
    Обновить last_avatar_url и last_followers_count только для SocialAccount, у которых есть хотя бы один пост с упоминанием проекта с is_leaderboard_project=True.
    Для каждого аккаунта используется только самый свежий пост (по posted_at).
    """
    from sqlalchemy import func, and_
    from sqlalchemy.orm import aliased
    session = SessionLocal()
    updated = 0
    try:
        latest_post_subq = (
            session.query(
                models.SocialPost.account_id.label('account_id'),
                func.max(models.SocialPost.posted_at).label('max_posted_at')
            )
            .group_by(models.SocialPost.account_id)
            .subquery()
        )
        LatestPost = aliased(models.SocialPost)
        results = (
            session.query(models.SocialAccount, LatestPost)
            .join(latest_post_subq, latest_post_subq.c.account_id == models.SocialAccount.id)
            .join(LatestPost, and_(
                LatestPost.account_id == models.SocialAccount.id,
                LatestPost.posted_at == latest_post_subq.c.max_posted_at
            ))
            .join(models.ProjectMention, LatestPost.id == models.ProjectMention.post_id)
            .join(models.Project, models.ProjectMention.project_id == models.Project.id)
            .filter(models.Project.is_leaderboard_project == True)
            .distinct()
            .all()
        )
        for account, latest_post in results:
            # Извлечь данные через extract_profile_info_from_post
            profile_url, followers_count = utils.extract_profile_info_from_post(latest_post)
            changed = False
            if profile_url and profile_url != account.last_avatar_url:
                account.last_avatar_url = profile_url
                changed = True
            if followers_count is not None and followers_count != account.last_followers_count:
                account.last_followers_count = followers_count
                changed = True
            if changed:
                session.add(account)
                updated += 1
        session.commit()
        print(f"Обновлено {updated} аккаунтов из {len(results)}.")
    except Exception as e:
        session.rollback()
        print(f"Ошибка при обновлении профилей: {str(e)}")
    finally:
        session.close()


def check_duplicates(clean: bool = False):
    """
    Проверяет (и опционально удаляет) дубликаты аккаунтов и постов по social_id/social_login.
    """
    from src import crud
    session = SessionLocal()
    try:
        report = asyncio.run(crud.check_for_unique(session, delete=clean))
        if not clean:
            print("Найдены дубликаты:")
        else:
            print("Удаление дубликатов завершено. Отчёт:")
        if not report["accounts"] and not report["posts"]:
            print("Дубликаты не найдены.")
            return
        if report["accounts"]:
            print("\nДубликаты аккаунтов:")
            for acc in report["accounts"]:
                print(f"  Тип: {acc['type']}, значение: {acc['value']}, оставлен id: {acc['keep']}, удалены id: {acc['delete']}")
        if report["posts"]:
            print("\nДубликаты постов:")
            for post in report["posts"]:
                print(f"  social_id: {post['social_id']}, оставлен id: {post['keep']}, удалены id: {post['delete']}")
    finally:
        session.close()


def refresh_leaderboard_cache():
    """
    Обновить кеш лидерборда для всех проектов с is_leaderboard_project=True.
    """
    import asyncio
    session = SessionLocal()
    from src.utils import get_leaderboard
    from src import models
    from src.services.cache_service import LeaderboardPeriod
    try:
        projects = session.query(models.Project).filter(models.Project.is_leaderboard_project == True).all()
        if not projects:
            print("Нет проектов с is_leaderboard_project=True")
            return
        print(f"Обновление кеша лидерборда для {len(projects)} проектов...")
        for project in projects:
            print(f"  - {project.name} (id={project.id}) ...", end=" ")
            try:
                # Обновляем кеш только для ALL_TIME (этого достаточно)
                asyncio.run(get_leaderboard(project, LeaderboardPeriod.ALL_TIME, session, force_rebuild=True))
                print("OK")
            except Exception as e:
                print(f"Ошибка: {e}")
        print("Обновление кеша завершено.")
    finally:
        session.close()


def get_arbus_score(handle: str, api_key: str):
    """
    Получить Arbus AI influence score для Twitter-аккаунта
    """
    url = f"https://api.arbus.ai/v1/arbus-score"
    params = {
        "twitter_handle": handle,
        "key": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        try:
            data = response.json()
        except Exception:
            print(f"Ошибка: не удалось декодировать ответ API как JSON. Код: {response.status_code}")
            print(response.text)
            return
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Ошибка при запросе к Arbus API: {e}")


def calculate_aura_for_top_posts():
    """
    Выбирает топ 5 постов за последний день для всех проектов с лидербордом,
    отправляет их на оценку в AI (Aura) и выводит результат.
    """
    session = SessionLocal()
    from src.services import aura_service
    from src import schemas
    from src.config import aura
    from typing import List

    NUMBER_OF_RUNS = 2

    try:
        print("Поиск проектов с лидербордом...")
        leaderboard_projects = session.query(models.Project).filter(models.Project.is_leaderboard_project == True).all()

        if not leaderboard_projects:
            print("Проекты с включенным лидербордом не найдены.")
            return

        print(f"Найдено {len(leaderboard_projects)} проектов. Сбор топ-5 постов для каждого...")

        all_posts_for_aura: List[schemas.PostForAura] = []
        temp_id_to_details_map = {}
        temp_id_counter = 1

        for project in leaderboard_projects:
            print(f"  - Проект: {project.name}")
            top_posts = crud.get_top_engagement_posts(project, session, limit=5, period=86400)
            if not top_posts:
                print(f"    -> Посты за последний день не найдены.")
                continue
            
            print(f"    -> Найдено {len(top_posts)} постов.")
            for post in top_posts:
                try:
                    full_text = utils_base.extract_full_post_text(post.raw_data, post.text)
                    
                    temp_id = temp_id_counter
                    all_posts_for_aura.append(schemas.PostForAura(id=temp_id, text=full_text))
                    
                    temp_id_to_details_map[temp_id] = {
                        'real_id': post.id,
                        'text': full_text,
                        'project_name': project.name
                    }
                    temp_id_counter += 1
                except PostTextExtractionError as e:
                    print(f"    -> [ПРЕДУПРЕЖДЕНИЕ] Не удалось извлечь текст для поста ID {post.id}. Ошибка: {e}")
                    continue
        
        if not all_posts_for_aura:
            print("Не найдено постов для анализа.")
            return

        print(f"\nВсего собрано {len(all_posts_for_aura)} постов. Запуск {NUMBER_OF_RUNS} раундов оценки для проверки стабильности...")
        
        all_runs_results = []
        for i in range(NUMBER_OF_RUNS):
            print(f"  - Запуск раунда {i+1}/{NUMBER_OF_RUNS}...")
            try:
                aura_results = asyncio.run(aura_service.ask_ai_to_calculate_posts_aura_score(all_posts_for_aura))
                if aura_results and aura_results.posts_scores:
                    all_runs_results.append(aura_results.posts_scores)
                else:
                    print(f"    -> [ПРЕДУПРЕЖДЕНИЕ] Раунд {i+1} не вернул оценок.")
            except Exception as e:
                 print(f"    -> [ОШИБКА] Раунд {i+1} не удался: {e}")

        if not all_runs_results:
            print("Не удалось получить оценки ни в одном из раундов. Прерывание.")
            return

        print("Все раунды завершены. Агрегация и расчет результатов...")

        aggregated_scores = defaultdict(lambda: defaultdict(list))
        for run_result in all_runs_results:
            for score_item in run_result:
                temp_id = score_item.post_id
                for criterion_id, score_value in score_item.scores.model_dump().items():
                    aggregated_scores[temp_id][criterion_id].append(score_value)

        scores_by_project = defaultdict(list)
        for temp_id in aggregated_scores:
             if temp_id not in temp_id_to_details_map:
                continue
             project_name = temp_id_to_details_map[temp_id]['project_name']
             scores_by_project[project_name].append(temp_id)

        for project_name, temp_ids in scores_by_project.items():
            print("\n" + "="*80)
            print(f"Проект: {project_name}")
            print("="*80)

            for temp_id in temp_ids:
                post_details = temp_id_to_details_map.get(temp_id)
                if not post_details:
                    continue

                real_post_id = post_details['real_id']
                post_text = post_details['text']
                
                print(f"\n--- Пост (Локальный ID: {temp_id}, DB ID: {real_post_id}) ---")
                print(f"Текст: {post_text}")
                print("\nОценки Aura (анализ стабильности):")
                
                total_aura_points = 0.0
                post_criteria_scores = aggregated_scores[temp_id]

                for criterion_info in aura.criteria_list:
                    criterion_id = criterion_info['id']
                    
                    if criterion_id not in post_criteria_scores:
                        continue

                    scores_list = post_criteria_scores[criterion_id]
                    average_score = sum(scores_list) / len(scores_list) if scores_list else 0
                    
                    criterion_name = criterion_info['categoria']
                    max_points = criterion_info['max_points']
                    
                    aura_points = (average_score / 10.0) * max_points
                    total_aura_points += aura_points
                    
                    scores_str = ", ".join(map(str, scores_list))
                    
                    print(f"  - {criterion_name} ({criterion_id}):")
                    print(f"    - Оценки: [{scores_str}] (среднее: {average_score:.2f}/10)")
                    print(f"    - Итоговые очки: {aura_points:.2f}/{max_points} Aura")

                total_max_aura_points = sum(c['max_points'] for c in aura.criteria_list)
                print(f"\n  Суммарная оценка Aura (на основе среднего): {total_aura_points:.2f}/{total_max_aura_points} Aura")
                print("-"*(len(str(real_post_id)) + 30))

    finally:
        session.close()


def export_leaderboard(project_id: int, output_path: str):
    """
    Экспортировать лидерборд проекта в Excel-файл.
    """
    session = SessionLocal()
    try:
        print(f"Экспорт лидерборда для проекта с ID={project_id} в файл {output_path}...")
        asyncio.run(utils.create_leaderboard_excel(project_id, session, output_path))
        print("Экспорт успешно завершен.")
    except Exception as e:
        print(f"Ошибка при экспорте лидерборда: {e}")
    finally:
        session.close()


def export_season_one_winners_csv():
    """
    Экспорт данных победителей первого сезона в формате CSV.
    Выводит на консоль: login, is_connected, user_id, "list of addresses with tx counts"
    """
    from src.config.season_one_winners import winners
    from src.services.web3_service import Web3Service
    from src.config.settings import settings
    from web3 import Web3
    from eth_utils import to_checksum_address
    
    session = SessionLocal()
    try:
        # Парсим победителей из строки
        winner_logins = [login.strip() for login in winners.strip().split('\n') if login.strip()]
        
        print("login,is_connected,user_id,addresses")
        
        # Инициализируем Web3Service для получения количества транзакций
        web3_service = Web3Service(settings.ANKR_API_KEY)
        
        for login in winner_logins:
            try:
                # Ищем пользователя по twitter_login
                user = session.execute(
                    select(User).where(User.twitter_login == login)
                ).scalar_one_or_none()
                
                # Ищем социальный аккаунт по social_login
                social_account = session.execute(
                    select(models.SocialAccount).where(models.SocialAccount.social_login == login)
                ).scalar_one_or_none()
                
                # Определяем is_connected
                is_connected = False
                if social_account and social_account.twitter_scout_score_updated_at is not None:
                    is_connected = True
                
                # Получаем user_id и адреса
                user_id = user.id if user else ""
                address_info_list = []
                
                if user and user.wallet_addresses:
                    # Сортируем адреса по дате добавления (от старых к новым)
                    sorted_wallets = sorted(user.wallet_addresses, key=lambda w: w.created_at)
                    for wallet in sorted_wallets:
                        tx_count = 0
                        formatted_address = wallet.address
                        
                        try:
                            if wallet.chain_family == enums.ChainFamily.EVM:
                                # Для Ethereum используем checksum формат
                                formatted_address = to_checksum_address(wallet.address)
                                
                                # Получаем количество транзакций для EVM адресов
                                # Пробуем разные сети
                                for chain_id in [enums.ChainID.ETHEREUM, enums.ChainID.BASE, enums.ChainID.ARBITRUM]:
                                    try:
                                        rpc_url = web3_service._get_rpc_http_url(chain_id)
                                        w3 = Web3(Web3.HTTPProvider(rpc_url))
                                        chain_tx_count = w3.eth.get_transaction_count(formatted_address)
                                        tx_count = max(tx_count, chain_tx_count)  # Берем максимальное количество транзакций
                                    except Exception as e:
                                        logger.debug(f"Не удалось получить tx count для {formatted_address} на сети {chain_id}: {e}")
                                        continue
                            elif wallet.chain_family == enums.ChainFamily.SOLANA:
                                # Для Solana оставляем адрес как есть
                                formatted_address = wallet.address
                                # Пропускаем получение транзакций для Solana (можно добавить позже если нужно)
                                
                        except Exception as e:
                            logger.error(f"Ошибка при форматировании адреса {wallet.address}: {str(e)}")
                            formatted_address = wallet.address
                        
                        # Формируем строку: address[tx_count]
                        address_info = f"{formatted_address} [{tx_count}]"
                        address_info_list.append(address_info)
                
                # Формируем строку с адресами
                addresses_str = ",".join(address_info_list) if address_info_list else ""
                
                # Выводим в формате CSV без кавычек
                print(f'{login},{str(is_connected).lower()},{user_id},{addresses_str}')
                
            except Exception as e:
                logger.error(f"Ошибка при обработке логина {login}: {str(e)}")
                # Выводим строку с ошибкой, но продолжаем обработку
                print(f'{login},false,,')
                continue
        
    finally:
        session.close()


def calculate_daily_aura(project_id: int = None):
    """
    Рассчитывает Aura score для постов за сутки для указанного проекта или всех проектов с лидербордом.
    
    Args:
        project_id: ID проекта для расчета. Если не указан, обрабатываются все проекты с лидербордом.
    """
    session = SessionLocal()
    try:
        if project_id is not None:
            project = session.query(Project).filter(Project.id == project_id).first()
            if not project:
                print(f"Проект с ID={project_id} не найден")
                return
                
            print(f"Расчет Aura score для проекта: {project.name} (ID={project_id})")
            processed = asyncio.run(utils.calculate_daily_aura_score(project, session))
            print(f"Обработано {processed} постов для проекта {project.name}")
        else:
            # Если ID не указан, обрабатываем все проекты с лидербордом
            projects = session.query(Project).filter(Project.is_leaderboard_project == True).all()
            if not projects:
                print("Не найдено проектов с включенным лидербордом")
                return
                
            total_processed = 0
            for project in projects:
                print(f"Расчет Aura score для проекта: {project.name} (ID={project.id})")
                processed = asyncio.run(utils.calculate_daily_aura_score(project, session))
                if processed:
                    total_processed += processed
                    print(f"Обработано {processed} постов")
                else:
                    print("Посты не найдены или не обработаны")
                    
            print(f"Всего обработано {total_processed} постов для {len(projects)} проектов")
    except Exception as e:
        print(f"Произошла ошибка при расчете Aura score: {e}")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='Утилиты для управления базой данных')
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # Команда удаления всех пользователей
    delete_parser = subparsers.add_parser('delete-all-users', help='Удалить всех пользователей')
    delete_parser.add_argument('--force', '-f', action='store_true', help='Пропустить подтверждение')

    # Команда создания промо-кода
    create_parser = subparsers.add_parser('create-promo-code', help='Создать промо-код')
    create_parser.add_argument('--code', '-c', required=True, help='Код промо-кода')
    create_parser.add_argument('--valid-until', '-v', required=True, help='Время окончания действия промо-кода')

    # Команда изменения промо-кода
    change_parser = subparsers.add_parser('change-promo-code', help='Изменить промо-код')
    change_parser.add_argument('--code', '-c', required=True, help='Код промо-кода')
    change_parser.add_argument('--valid-until', '-v', required=True, help='Время окончания действия промо-кода')

    # Команда синхронизации проектов
    sync_parser = subparsers.add_parser('sync-projects', help='Синхронизировать проекты')

    # Команда синхронизации постов
    sync_posts_parser = subparsers.add_parser('sync-posts', help='Синхронизировать посты')

    # Команда очистки старых постов
    cleanup_parser = subparsers.add_parser('cleanup-old-posts', help='Удалить старые посты')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Показать что будет удалено, но не удалять')

    # Команда создания админа/модератора
    create_admin_parser = subparsers.add_parser('create-admin', help='Создать администратора/модератора')
    create_admin_parser.add_argument('--login', '-l', required=True, help='Логин')
    create_admin_parser.add_argument('--password', '-p', required=True, help='Пароль')
    create_admin_parser.add_argument('--role', '-r', required=False, default='admin', help='Роль (admin|moderator)')

    # Команда удаления всех админов
    delete_admins_parser = subparsers.add_parser('delete-all-admins', help='Удалить всех администраторов')
    delete_admins_parser.add_argument('--force', '-f', action='store_true', help='Пропустить подтверждение')

    # Команда обновления лидерборда
    leaderboard_parser = subparsers.add_parser('update-leaderboard', help='Обновить лидерборд для проекта или всех проектов')
    leaderboard_parser.add_argument('--project-id', '-p', type=int, required=False, help='ID проекта (если не указан, обновляются все проекты с is_leaderboard_project=True)')

    # Команда мастер-апдейта
    master_update_parser = subparsers.add_parser('master-update', help='Мастер-обновление: синк постов, лидерборд, очистка')

    # Команда обновления xscore пользователей
    update_xscore_parser = subparsers.add_parser('update-users-xscore', help='Обновить xscore для всех пользователей, чьи аккаунты упоминают лидербордские проекты')

    # Команда создания auto-yaps
    create_auto_yaps_parser = subparsers.add_parser('create-auto-yaps', help='Создать auto-yaps для проекта или всех проектов')
    create_auto_yaps_parser.add_argument('-n', '--name', required=False, help='Название проекта (если не указано, для всех проектов)')

    # Команда обновления профиля соц. аккаунтов
    update_social_accounts_profile_parser = subparsers.add_parser('update-social-accounts-profile', help='Обновить last_avatar_url и last_followers_count для всех SocialAccount на основе самого свежего поста')

    # Команда поиска и чистки дубликатов
    check_dupes_parser = subparsers.add_parser('check-duplicates', help='Проверить и/или удалить дубликаты аккаунтов и постов')
    check_dupes_parser.add_argument('--clean', action='store_true', help='Удалить найденные дубликаты')

    # Команда обновления кеша лидерборда для всех проектов
    refresh_leaderboard_cache_parser = subparsers.add_parser('refresh-leaderboard-cache', help='Обновить кеш лидерборда для всех проектов с is_leaderboard_project=True')

    # Команда получения Arbus Score
    arbus_score_parser = subparsers.add_parser('get-arbus-score', help='Получить Arbus AI influence score для Twitter-аккаунта')
    arbus_score_parser.add_argument('--handle', '-l', required=True, help='Twitter handle (ник без @)')

    # Команда оценки постов через Aura
    aura_score_parser = subparsers.add_parser('calculate-aura', help='Оценить топ-5 постов для проектов с лидербордом')

    # Команда экспорта лидерборда в Excel
    export_leaderboard_parser = subparsers.add_parser('export-leaderboard', help='Экспортировать лидерборд проекта в Excel')
    export_leaderboard_parser.add_argument('--project-id', '-p', type=int, required=True, help='ID проекта')
    export_leaderboard_parser.add_argument('--output-path', '-o', type=str, required=True, help='Путь для сохранения Excel файла')

    # Команда экспорта победителей первого сезона в CSV
    export_season_winners_parser = subparsers.add_parser('export-season-winners', help='Экспортировать данные победителей первого сезона в CSV')

    # Команда ежедневного расчета Aura score
    daily_aura_parser = subparsers.add_parser('calculate-daily-aura', help='Рассчитать Aura score для постов за сутки')
    daily_aura_parser.add_argument('--project-id', '-p', type=int, required=False, help='ID проекта (если не указан, обрабатываются все проекты с лидербордом)')


    args = parser.parse_args()

    if args.command == 'delete-all-users':
        delete_all_users(args.force)
    elif args.command == 'create-promo-code':
        create_promo_code(args.code, args.valid_until)
    elif args.command == 'change-promo-code':
        change_promo_code(args.code, args.valid_until)
    elif args.command == 'sync-projects':
        sync_projects()
    elif args.command == 'sync-posts':
        sync_posts()
    elif args.command == 'cleanup-old-posts':
        cleanup_old_posts(args.dry_run)
    elif args.command == 'create-admin':
        create_admin_user(args.login, args.password, args.role)
    elif args.command == 'delete-all-admins':
        delete_all_admin_users(args.force)
    elif args.command == 'update-leaderboard':
        update_leaderboard(args.project_id)
    elif args.command == 'master-update':
        master_update()
    elif args.command == 'update-users-xscore':
        update_users_xscore()
    elif args.command == 'create-auto-yaps':
        create_auto_yaps(args.name)
    elif args.command == 'update-social-accounts-profile':
        update_social_accounts_profile()
    elif args.command == 'check-duplicates':
        check_duplicates(args.clean)
    elif args.command == 'refresh-leaderboard-cache':
        refresh_leaderboard_cache()
    elif args.command == 'get-arbus-score':
        get_arbus_score(args.handle, "AIzaSyDkiIG4QdLvYsSzlPMh238BwGZtQZQjop0")
    elif args.command == 'calculate-aura':
        calculate_aura_for_top_posts()
    elif args.command == 'export-leaderboard':
        export_leaderboard(args.project_id, args.output_path)
    elif args.command == 'export-season-winners':
        export_season_one_winners_csv()
    elif args.command == 'calculate-daily-aura':
        calculate_daily_aura(args.project_id)
    else:
        parser.print_help()


"""
python -m src.cli delete-all-users --force
python -m src.cli create-promo-code --code "PROMO_CODE" --valid-until "VALID_UNTIL"
python -m src.cli change-promo-code --code "PROMO_CODE" --valid-until "VALID_UNTIL"
python -m src.cli sync-projects
python -m src.cli sync-posts
python -m src.cli cleanup-old-posts --dry-run
python -m src.cli cleanup-old-posts
python -m src.cli create-admin --login "LOGIN" --password "PASSWORD" --role "ROLE"
python -m src.cli master-update
python -m src.cli update-users-xscore
python -m src.cli update-leaderboard
python -m src.cli create-auto-yaps -n "ProjectName"
python -m src.cli create-auto-yaps
python -m src.cli update-social-accounts-profile
python -m src.cli check-duplicates --clean
python -m src.cli refresh-leaderboard-cache
python -m src.cli get-arbus-score -l "HANDLE"
python -m src.cli calculate-aura
python -m src.cli export-leaderboard -p 1 -o leaderboard.xlsx
python -m src.cli export-season-winners
python -m src.cli calculate-daily-aura
python -m src.cli calculate-daily-aura -p 1
"""

if __name__ == "__main__":
    main()

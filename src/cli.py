import asyncio
import argparse
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.database import SessionLocal
from src.models import User, Project, AdminUser
from src import crud, utils_base, enums, models, utils
from src.config.projects import projects_all
import logging
import bcrypt

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
        from_timestamp = utils_base.now_timestamp() - 3 * 24 * 60 * 60
        # Для каждого проекта запускаем синхронизацию постов
        for project in projects:
            await utils.update_project_data(project, from_timestamp, session)
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
        cutoff_timestamp = utils_base.now_timestamp() - settings.POST_INACTIVE_TIME_SECONDS
        
        if dry_run:
            # В режиме dry-run только показываем, что будет удалено
            stmt = select(models.SocialPost).where(models.SocialPost.posted_at < cutoff_timestamp)
            posts_to_delete = session.execute(stmt).scalars().all()
            print(f"В режиме dry-run: найдено {len(posts_to_delete)} постов для удаления")
            print(f"Cutoff timestamp: {cutoff_timestamp} (посты старше {settings.POST_INACTIVE_TIME_SECONDS} секунд)")
            for post in posts_to_delete[:5]:  # Показываем первые 5 для примера
                print(f"  - Post ID: {post.id}, Posted at: {post.posted_at}, Text: {post.text[:50]}...")
            if len(posts_to_delete) > 5:
                print(f"  ... и ещё {len(posts_to_delete) - 5} постов")
        else:
            # Реальное удаление
            deleted_count = await crud.delete_old_posts(session, cutoff_timestamp)
            print(f"Успешно удалено {deleted_count} старых постов")
            print(f"Cutoff timestamp: {cutoff_timestamp} (посты старше {settings.POST_INACTIVE_TIME_SECONDS} секунд)")
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
"""

if __name__ == "__main__":
    main()

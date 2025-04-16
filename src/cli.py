import asyncio
import argparse
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.database import SessionLocal
from src.models import User
from src import crud


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
                asyncio.run(crud.delete_user(user.id, session))
                deleted_count += 1
            except Exception as e:
                print(f"Ошибка при удалении пользователя {user.id}: {str(e)}")
                continue
        
        print(f"Успешно удалено {deleted_count} пользователей")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='Утилиты для управления базой данных')
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # Команда удаления всех пользователей
    delete_parser = subparsers.add_parser('delete-all-users', help='Удалить всех пользователей')
    delete_parser.add_argument('--force', '-f', action='store_true', help='Пропустить подтверждение')

    args = parser.parse_args()

    if args.command == 'delete-all-users':
        delete_all_users(args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

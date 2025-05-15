import asyncio
import argparse
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.database import SessionLocal
from src.models import User
from src import crud, utils_base


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

    args = parser.parse_args()

    if args.command == 'delete-all-users':
        delete_all_users(args.force)
    elif args.command == 'create-promo-code':
        create_promo_code(args.code, args.valid_until)
    elif args.command == 'change-promo-code':
        change_promo_code(args.code, args.valid_until)
    else:
        parser.print_help()


"""
python -m src.cli delete-all-users --force
python -m src.cli create-promo-code --code "PROMO_CODE" --valid-until "VALID_UNTIL"
python -m src.cli change-promo-code --code "PROMO_CODE" --valid-until "VALID_UNTIL"
"""

if __name__ == "__main__":
    main()

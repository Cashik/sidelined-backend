CLI команды

Удаление юзеров:
python -m src.cli delete-all-users 

Редактирование промо кодов:
python -m src.cli create-promo-code --code "PROMO_CODE" --valid-until {timestamp}
python -m src.cli change-promo-code --code "PROMO_CODE" --valid-until {timestamp}
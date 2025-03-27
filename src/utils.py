import time
import json
from typing import Optional, Dict, List, Any

import openai
from openai import OpenAI

from src import schemas, enums, models, exceptions
from src.config import settings
from src.services import thirdweb_service

def now_timestamp():
    """Получение текущего timestamp в секундах"""
    return int(time.time())


async def get_ai_answer(chat: schemas.Chat, model: enums.Model) -> schemas.Message:
    """
    Получение ответа от ИИ на основе контекста чата
    
    Args:
        chat: Текущий чат со всеми сообщениями
        model: Модель ИИ для генерации ответа
    
    Returns:
        Сгенерированное сообщение от ИИ
    """
    # Инициализация клиента OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Формирование истории сообщений для запроса к API
    messages = []
    
    # Системное сообщение (инструкции для модели)
    messages.append({
        "role": "system",
        "content": "You are a helpful assistant."
    })
    
    # Добавление сообщений из истории чата
    # Проходим по всем nonce в порядке возрастания
    for nonce in sorted(chat.messages.keys()):
        # Выбираем сообщение с наибольшим selected_at
        message = max(chat.messages[nonce], key=lambda x: x.selected_at)
        messages.append({
            "role": message.sender.value,
            "content": message.content
        })
    
    try:
        # Отправка запроса к OpenAI
        model_name = model.value  # Получаем строковое значение из enum
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        # Получение ответа
        content = response.choices[0].message.content
        
        # Определение нового nonce (следующий после последнего в чате)
        last_nonce = max(chat.messages.keys()) if chat.messages else -1
        new_nonce = last_nonce + 1
        
        # Создание объекта сообщения
        answer_message = schemas.Message(
            content=content,
            sender=enums.Role.ASSISTANT,
            recipient=enums.Role.USER,
            model=model,
            nonce=new_nonce,
            created_at=now_timestamp(),
            selected_at=now_timestamp()
        )
        
        return answer_message
        
    except Exception as e:
        # В случае ошибки создаем сообщение с текстом ошибки
        if settings.DEBUG:
            error_message = f"Error while generating answer: {str(e)}"
        else:
            error_message = "Error while generating answer. Please try again later."
        
        # Определение нового nonce
        last_nonce = max(chat.messages.keys()) if chat.messages else -1
        new_nonce = last_nonce + 1
        
        # Создание объекта сообщения с ошибкой
        answer_message = schemas.Message(
            content=error_message,
            sender=enums.Role.SYSTEM,
            recipient=enums.Role.USER,
            model=model,
            nonce=new_nonce,
            created_at=now_timestamp(),
            selected_at=now_timestamp()
        )
        
        return answer_message


async def check_user_access(user_address: str, user_chain_id: int) -> bool:
    """
    Проверка баланса пользователя в Thirdweb
    """
    # для начала отсеиваем все требования по цепочкам, кроме той, что указана в user_chain_id
    requirements = [req for req in settings.TOKEN_REQUIREMENTS if req.token.chain_id.value == user_chain_id]
    
    if not requirements:
        return False
    
    # какие-то требования остались, значит нужно проверить балансы
    
    thirdweb = thirdweb_service.ThirdwebService(
        app_id=settings.THIRDWEB_APP_ID,
        private_key=settings.THIRDWEB_PRIVATE_KEY
    )
    
    # получаем балансы всех токенов
    try:
        balances: list[thirdweb_service.TokenBalance] = await thirdweb.get_balances(user_address, user_chain_id, enums.TokenInterface.ERC20)
    except exceptions.ThirdwebServiceException:
        return settings.ALLOW_CHAT_WHEN_SERVER_IS_DOWN
    
    # далее проходим по всем требованиям и проверяем баланс
    for req in requirements:
        for balance in balances:
            if balance.address_lower == req.token.address.lower() and balance.chain_id == user_chain_id:
                actual_balance = balance.balance//(10**req.token.decimals)
                if actual_balance >= req.balance:
                    return True
    
    # ничего не подошло, значит баланс не соответствует ни одному требованию
    return False
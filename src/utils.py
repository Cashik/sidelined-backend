import time
import json
from typing import Optional, Dict, List, Any
import logging
import openai
from openai import OpenAI

from src import schemas, enums, models, exceptions
from src.config import settings
from src.services import thirdweb_service

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def now_timestamp():
    """Получение текущего timestamp в секундах"""
    return int(time.time())


async def get_ai_answer(generate_data: schemas.AssistantGenerateData) -> schemas.Message:
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
    system_message = f"You are a helpful assistant."
    if generate_data.user.preferred_name:
        system_message += f"\nUser wants to be called as {generate_data.user.preferred_name}."
    if generate_data.user.user_context:
        system_message += f"\nUser context: {generate_data.user.user_context}."
    if generate_data.chat_settings.chat_style:
        system_message += f"\nUse this style for your answers: {generate_data.chat_settings.chat_style.value}."
    if generate_data.chat_settings.chat_details_level:
        system_message += f"\nUse this details level for your answers: {generate_data.chat_settings.chat_details_level.value}."
    if generate_data.user.facts:
        system_message += f"\nWhat you know about user:"
        for fact in generate_data.user.facts:
            system_message += f"\n- {fact.description}"
    logger.info(f"System message: {system_message}")
    messages.append({
        "role": "system",
        "content": system_message
    })
    
    # Добавление сообщений из истории чата
    # Проходим по всем nonce в порядке возрастания
    for nonce in sorted(generate_data.chat.messages.keys()):
        # Выбираем сообщение с наибольшим selected_at
        message = max(generate_data.chat.messages[nonce], key=lambda x: x.selected_at)
        messages.append({
            "role": message.sender.value,
            "content": message.content
        })
    
    try:
        # Отправка запроса к OpenAI
        model_name = generate_data.chat_settings.model.value  # Получаем строковое значение из enum
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        # Получение ответа
        content = response.choices[0].message.content
        
        # Определение нового nonce (следующий после последнего в чате)
        last_nonce = max(generate_data.chat.messages.keys()) if generate_data.chat.messages else -1
        new_nonce = last_nonce + 1
        
        # Создание объекта сообщения
        answer_message = schemas.Message(
            content=content,
            sender=enums.Role.ASSISTANT,
            recipient=enums.Role.USER,
            model=generate_data.chat_settings.model,
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
        last_nonce = max(generate_data.chat.messages.keys()) if generate_data.chat.messages else -1
        new_nonce = last_nonce + 1
        
        # Создание объекта сообщения с ошибкой
        answer_message = schemas.Message(
            content=error_message,
            sender=enums.Role.SYSTEM,
            recipient=enums.Role.USER,
            model=generate_data.chat_settings.model,
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
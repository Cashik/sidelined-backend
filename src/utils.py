import time
import json
from typing import Optional, Dict, List, Any

import openai
from openai import OpenAI

from src import schemas, enums, models
from src.config import settings


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

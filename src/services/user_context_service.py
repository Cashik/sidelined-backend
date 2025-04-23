import logging
from typing import List, Dict, Any, Optional
import json
import time
from sqlalchemy import select, desc, func
from sqlalchemy.orm import Session
import asyncio
from datetime import datetime

from src import models, crud, schemas, enums, utils
from src.config import settings
from src.database import get_session
from src.providers.gemini import GeminiProvider, GeminiNotesResponse
logger = logging.getLogger(__name__)


async def test_background_task(user_id: int, message: str):
    """
    Тестовая фоновая задача для проверки работы BackgroundTasks.
    
    Args:
        user_id: ID пользователя
        message: Тестовое сообщение
    """
    # Задержка для имитации длительной работы
    await asyncio.sleep(5)
    
    logger.info(f"Test background task executed: user_id={user_id}, message={message}")
    
    # Используем новую сессию
    session = next(get_session())
    try:
        # Здесь можно добавить тестовый факт, чтобы проверить работу
        test_facts = [f"Тестовый факт: {message} (создан в {utils.now_timestamp()})"]
        await crud.add_user_facts(user_id, test_facts, session)
        logger.info(f"Added test fact for user_id={user_id}")
    except Exception as e:
        logger.error(f"Error in test background task: {str(e)}")
    finally:
        session.close() 

async def update_user_information(user_id: int):
    """
    Анализирует все новые сообщения пользователя из всех чатов, делает запрос к Gemini
    и обновляет информацию о пользователе на основе полученного ответа.
    
    Args:
        user_id: ID пользователя
    
    Returns:
        Dict с результатами обновления информации пользователя
    """
    logger.info(f"Starting user information update for user_id={user_id}")
    
    session = next(get_session())
    result = {
        "success": False,
        "deleted_notes": [],
        "added_notes": [],
        "processed_chats": [],
        "error": None
    }
    
    if not settings.FACTS_FUNCTIONALITY_ENABLED:
        result["success"] = True
        logger.info(f"User context functionality is disabled.")
        return result
    
    try:
        # Получаем пользователя
        user = await crud.get_user_by_id(user_id, session)
        if not user:
            result["error"] = f"User not found: user_id={user_id}"
            logger.error(result["error"])
            return result
        
        # Проверяем, есть ли новые сообщения для анализа
        messages_to_analyze = []
        
        # Получаем из базы все сообщения пользователя, которые не были проанализированы
        messages_to_analyze = await crud.get_user_messages_to_analyze(user_id, session)
        
        # TODO: продумать значение минимального количества сообщений для анализа или придумать что-то другое для фильтра проверки
        if not messages_to_analyze or len(messages_to_analyze) < 1:
            result["success"] = True
            logger.info(f"Not enough messages to analyze for user_id={user_id}")
            return result
        
        logger.info(f"Found {len(messages_to_analyze)} messages to analyze for user_id={user_id}")
        
        # Создаем системный промпт для модели
        system_prompt = create_system_prompt(user, messages_to_analyze)
        
        # Отправляем запрос к модели Gemini
        gemini_provider = GeminiProvider()
        try:
            gemini_response: GeminiNotesResponse = await gemini_provider.generate_notes_response(system_prompt)
        except Exception as e:
            result["error"] = f"Failed to get response from Gemini model: {str(e)}"
            logger.error(result["error"])
            return result
        
        # Удаляем выбранные заметки
        if gemini_response.delete_notes:
            deleted_facts = await crud.delete_user_facts(user_id, gemini_response.delete_notes, session)
            result["deleted_notes"] = deleted_facts
        
        # Добавляем новые заметки
        if gemini_response.new_notes:
            _, added_facts = await crud.add_user_facts(user_id, gemini_response.new_notes, session)
            result["added_notes"] = added_facts
        
        # Обновляем last_analysed_nonce для всех чатов с новыми сообщениями
        # собираем набор всех чатов из проанализированных сообщений
        chats_to_update = dict() 
        for message in messages_to_analyze:
            if message.chat_id not in chats_to_update:
                chats_to_update[message.chat_id] = message.nonce
            else:
                chats_to_update[message.chat_id] = max(chats_to_update[message.chat_id], message.nonce)
        
        # обновляем last_analysed_nonce для всех чатов
        for chat_id, nonce in chats_to_update.items():
            chat = session.query(models.Chat).filter(models.Chat.id == chat_id).first()
            chat.last_analysed_nonce = nonce
            session.add(chat)
        
        # Сохраняем изменения
        session.commit()
        
        result["success"] = True
        logger.info(f"Successfully updated user information for user_id={user_id}")
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error updating user information: {str(e)}")
    finally:
        session.close()
    
    return result

def create_system_prompt(user: models.User, messages_to_analyze: List[models.Message]) -> str:
    """
    Создает системный промпт для модели Gemini на основе информации о пользователе.
    
    Args:
        user: Объект пользователя
        
    Returns:
        Строка с системным промптом
    """

    # Сортируем сообщения от старых к новым
    messages_to_analyze.sort(key=lambda x: x.created_at)
    
    # Форматируем сообщения в текст
    messages_text = ""
    for i, message in enumerate(messages_to_analyze):
        timestamp = datetime.fromtimestamp(message.created_at).strftime("%Y-%m-%d %H:%M:%S")
        messages_text += f"[{timestamp}] Message {i+1}: {message.content}\n\n"
    
    notes_text = ""
    for fact in user.facts:
        notes_text += f"{fact.id} - {fact.description}\n"
    if not notes_text:
        notes_text = "No existing notes about user."
    
    # Формируем промпт для Gemini
    prompt = f"""
    You are a analitics assistant that helps to update notes about user based on his messages.

    Your task is to analyze user's messages and collect any important information about him.
    
    Important information may be:
    - what user likes or dislikes (only important things)
    - information about his habits, preferences and etc.
    - circle of friends and important information about them
    - professional activity and achievements
    - personal qualities and character traits and external features
    - any other important facts that can help to answer user's questions better
    
    Not important information:
    - not important notes like "love sosiges with mayonez" or "user is a good person"
    - events or facts that are not important for future conversations like "user walks 5 km today" or "user see a new movie yesterday"
    - user's wallet addresses (because they are controlled by user and can be changed)
    
    Exaples of good and bad notes:
    - bad: "user like to listens rock music"
    - good: "likes rock music"
    - bad: "ate pizza today like every day"
    - good: "likes pizza"
    - bad: "has a friend Alex who constantly bullies him"
    - good: "Alex - friend", "Alex is a bully"
    - bad: "user has a girlfriend Marina who is a doctor"
    - good: "girlfriend Marina", "Marina is a doctor"

    EXISTING NOTES ABOUT USER(id - note text):
    {notes_text}
    
    USER'S MESSAGES TO ANALYZE:
    (messages sorted chronologically from oldest to newest from all user's chats)
    {messages_text}
    
    INSTRUCTIONS:
    1. Analyze user's messages and find any important information about him.
    2. Determine which of the existing notes are outdated or contradict the new information - they need to be deleted or changed (recreated).
    3. Try to make notes as short as possible. For example, not "user likes to listen to music", but "likes music".
    4. Do not repeat existing notes.
    5. Use only English for notes, even if the user uses other languages.
    6. Pay special attention to the most recent messages as they contain the most up-to-date information about the user.
    
    Return structured answer in JSON format with the following fields:
       - "delete_notes": list of note ids to delete
       - "new_notes": list of new notes to add
    
    Answer must contain only JSON, without additional explanations.
    """
    
    return prompt


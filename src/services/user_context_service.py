import logging
from typing import List, Dict, Any, Optional
import json
import time
from sqlalchemy import select, desc, func
from sqlalchemy.orm import Session
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field

from src import models, crud, schemas, enums, utils_base
from src.config.settings import settings
from src.database import get_session

logger = logging.getLogger(__name__)


class NotesResponse(BaseModel):
    """Структура ответа для анализа заметок о пользователе"""
    delete_notes: Optional[List[int]] = Field(default=None, description="List of note ids to delete")
    new_notes: Optional[List[str]] = Field(default=None, description="List of new notes to add")


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
        messages_to_analyze = await crud.get_user_messages_to_analyze(user, session)
        
        # TODO: продумать значение минимального количества сообщений для анализа или придумать что-то другое для фильтра проверки
        if not messages_to_analyze or len(messages_to_analyze) < 1:
            result["success"] = True
            logger.info(f"Not enough messages to analyze for user_id={user_id}")
            return result
        
        logger.info(f"Found {len(messages_to_analyze)} messages to analyze for user_id={user_id}")
        
        # Создаем системный промпт для модели
        system_prompt = create_system_prompt(user, messages_to_analyze)
        
        # Отправляем запрос к модели через langchain
        try:
            notes_response: NotesResponse = await generate_notes_analysis(system_prompt)
        except Exception as e:
            result["error"] = f"Failed to get response from AI model: {str(e)}"
            logger.error(result["error"])
            return result
        
        logger.info(f"Notes response: {notes_response}")
        
        # Удаляем выбранные заметки
        if notes_response.delete_notes:
            deleted_facts = await crud.delete_user_facts(user, notes_response.delete_notes, session)
            result["deleted_notes"] = deleted_facts
        
        # Добавляем новые заметки
        if notes_response.new_notes:
            _, added_facts = await crud.add_user_facts(user, notes_response.new_notes, session)
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


async def generate_notes_analysis(system_prompt: str) -> NotesResponse:
    """
    Генерирует анализ заметок о пользователе через langchain.
    
    Args:
        system_prompt: Системный промпт для анализа
        
    Returns:
        NotesResponse с результатами анализа
    """
    from langchain.chat_models import init_chat_model
    from pydantic import ValidationError
    
    # Определяем модель и провайдер (используем Gemini по умолчанию для анализа заметок)
    model = enums.Model.GEMINI_2_5_PRO
    model_provider = "google_genai"
    api_key = settings.GEMINI_API_KEY
    
    # Инициализация LLM с structured output
    llm = init_chat_model(
        model=model.value,
        model_provider=model_provider,
        api_key=api_key,
        temperature=0.1,  # Низкая температура для более точного анализа
        max_tokens=1024,
    ).with_structured_output(NotesResponse)
    
    try:
        result: NotesResponse = await llm.ainvoke([
            {"role": "user", "content": system_prompt}
        ])
        return result
    except ValidationError as e:
        logger.error(f"Validation error in notes analysis: {e}")
        # Повторная попытка с более строгим промптом
        strict_prompt = system_prompt + "\n\nIMPORTANT: Your response MUST be valid JSON that fully complies with the schema, otherwise an error will occur."
        result: NotesResponse = await llm.ainvoke([
            {"role": "user", "content": strict_prompt}
        ])
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


import re
import time
import json
from typing import Optional, Dict, List, Any
import logging
import openai
from openai import OpenAI
from sqlalchemy.orm import Session

from src import schemas, enums, models, exceptions, crud
from src.config import settings
from src.services import thirdweb_service
from src.services.prompt_service import PromptService
from src.providers import openai, gemini

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)





def now_timestamp():
    """Получение текущего timestamp в секундах"""
    return int(time.time())


async def handle_agent_functions(raw_message: str, user_id: int, db: Session) -> schemas.AgentFunctionCallingResult:
    """Обработка команд в тексте"""
    logger.info(f"Raw message: {raw_message}")
    # Найти все команды в тексте
    commands = re.findall(r'call\s+(\w+)\((.*?)\)', raw_message)
    messages = []
    
    # Инициализация сервиса Thirdweb
    thirdweb = thirdweb_service.ThirdwebService(
        app_id=settings.THIRDWEB_APP_ID,
        private_key=settings.THIRDWEB_PRIVATE_KEY
    )
    
    for command in commands:
        function_name = command[0]
        args = command[1]
        
        if function_name == "nebula_ask" and settings.NEBULA_FUNCTIONALITY_ENABLED:
            try:
                # Извлекаем строку из аргументов
                message = args.strip('"').strip("'")
                # Создаем запрос к Nebula
                request = thirdweb_service.NebulaChatRequest(message=message)
                # Получаем ответ
                response = await thirdweb.nebula_chat(request)
                messages.append(schemas.SystemMessage(
                    message=f"Nebula:\n {response.message}"
                ))
            except Exception as e:
                logger.error(f"Error asking Nebula: {e}")
                messages.append(schemas.SystemMessage(
                    message=f"Error asking Nebula: {str(e)}"
                ))
        elif function_name == "add_facts" and settings.FACTS_FUNCTIONALITY_ENABLED:
            try:
                facts: list[str] = json.loads(args)
                await crud.add_user_facts(user_id, facts, db)
            except Exception as e:
                logger.error(f"Error adding facts: {e}")
                messages.append(schemas.SystemMessage(
                    message=f"Error adding facts: {str(e)}"
                ))
        elif function_name == "del_facts" and settings.FACTS_FUNCTIONALITY_ENABLED:
            try:
                ids: list[int] = json.loads(args)
                await crud.delete_user_facts(user_id, ids, db)
            except Exception as e:
                logger.error(f"Error removing facts: {e}")
                messages.append(schemas.SystemMessage(
                    message=f"Error removing facts: {str(e)}"
                ))
        else:
            messages.append(schemas.SystemMessage(
                message=f"dev: agent called function {function_name} with arguments {args}"
            ))
    
    # Удаляем команды вместе с переносами строк
    edited_raw_message = re.sub(r'\n\s*call\s+(\w+)\((.*?)\)', '', raw_message)
    
    return schemas.AgentFunctionCallingResult(
        success=True,
        new_messages=messages,
        edited_raw_message=edited_raw_message
    )


async def get_ai_answer(generate_data: schemas.AssistantGenerateData, user_id: int, db: Session) -> List[schemas.Message]:
    """
    Получение ответа от ИИ на основе контекста чата
    """
    try:
        logger.info(f"Configuring prompt service ...")
        prompt_service = PromptService(generate_data)
        logger.info(f"Setting up AI provider ...")
        logger.info(f"Model: {generate_data.chat_settings.model} model value: {generate_data.chat_settings.model.value} gemini value: {enums.Model.GEMINI_2_FLASH.value}")
        if generate_data.chat_settings.model in (enums.Model.GPT_4, enums.Model.GPT_4O, enums.Model.GPT_4O_MINI) and settings.OPENAI_API_KEY:
            ai_provider = openai.OpenAIProvider()
        elif generate_data.chat_settings.model in (enums.Model.GEMINI_2_FLASH, enums.Model.GEMINI_2_5_PRO) and settings.GEMINI_API_KEY:
            ai_provider = gemini.GeminiProvider()
        else:
            raise NotImplementedError(f"Model \"{generate_data.chat_settings.model.value}\" is not implemented yet!")
        logger.info(f"Generating response ...")
        raw_answer = await ai_provider.generate_response(prompt_service)
        logger.info(f"Response generated: {raw_answer}")
        
        # Определение нового nonce (следующий после последнего в чате)
        last_nonce = max(generate_data.chat.messages.keys()) if generate_data.chat.messages else -1
        new_nonce = last_nonce + 1
        
        if settings.FUNCTIONALITY_ENABLED:
            # Обработка команд в тексте
            agent_function_calling_result = await handle_agent_functions(raw_answer, user_id, db)
        
        # Создание объекта сообщения
        result_messages = []
        # Добавляем новые сообщение от агента
        result_messages.append(schemas.Message(
            content=agent_function_calling_result.edited_raw_message,
            sender=enums.Role.ASSISTANT,
            recipient=enums.Role.USER,
            model=generate_data.chat_settings.model,
            nonce=new_nonce,
            created_at=now_timestamp(),
            selected_at=now_timestamp()
        ))
        # Если есть новые сообщения от системы, добавляем их
        for message in agent_function_calling_result.new_messages:
            new_nonce += 1
            result_messages.append(schemas.Message(
                content=message.message,
                sender=enums.Role.SYSTEM,
                recipient=enums.Role.USER,
                model=generate_data.chat_settings.model,
                nonce=new_nonce,
                created_at=now_timestamp(),
                selected_at=now_timestamp()
            ))
        
        return result_messages
        
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
        
        return [answer_message]


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
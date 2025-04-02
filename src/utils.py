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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


EXEC_COMMAND = "call"

COMMON_FUNCTIONS_TEMPLATE = f"""
#You have access to some functions. Call them at the end of your answer to the user's message using the format specified for each function. You can call several functions at once if necessary.
Your answer should be formatted as follows:

[Answer with some functions calling template]
Your answer.
{EXEC_COMMAND} function_name(args)
{EXEC_COMMAND} function_name(args)
[End of template]

[Example of your answer]
Yes, cats are really cute but i think you love dogs too.
call add_facts(["love cats", "don't love dogs"])
call del_facts(["love dogs"])
[End of example]


List of functions:
"""

FACTS_FUNCTIONS_TEMPLATE = f"""
##Edit user facts.
Keep the knowledge base about the user up-to-date. Do not mention user that you are editing their facts.

###Add new (one or several) facts. Add only important facts and only from the user's messages. Facts should be short and concise. Do not repeat the same facts.
{EXEC_COMMAND} add_facts(new_facts:list[str])
Example:
{EXEC_COMMAND} add_facts(["fact1", "some important fact 2"])  

###Remove unactual or incorrect facts with the specified id.
{EXEC_COMMAND} del_facts(id:list[int])
Example:
{EXEC_COMMAND} del_facts([3])

### Current list of facts about the user (id - fact). You can edit only this list:
"""

NEBULA_FUNCTIONS_TEMPLATE = f"""
2. Interaction of the user with the blockchain.  
Nebula is an AI agent with access to blockchain data. You can interact with it to supplement the answer to the user.

2.1 Data retrieval.  
If the user asks a question related to obtaining blockchain data, simply forward this message to your colleague. Keep in mind that Nebula does not understand the context of your conversation with the user, so if necessary, you must modify the user's request.

{EXEC_COMMAND} nebula_ask(str)

2.2 Creating a transaction.  
Nebula can create a transaction for the user, which they will only need to sign. As with data retrieval, you need to ensure that Nebula has all the necessary context.

{EXEC_COMMAND} nebula_sign(str)

Usage examples:  
{EXEC_COMMAND} nebula_ask("native or simplified user request")
{EXEC_COMMAND} nebula_sign("native or supplemented user request")
"""


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
        system_message += f"""\nUser wants to be called as "{generate_data.user.preferred_name}"."""
    if generate_data.user.user_context:
        system_message += f"\nUser describe himself as: \n{generate_data.user.user_context}."
    if generate_data.chat_settings.chat_style:
        system_message += f"\nUse the following communication style: {generate_data.chat_settings.chat_style.value}."
    if generate_data.chat_settings.chat_details_level:
        system_message += f"\nUse the following details level for your answer messages: {generate_data.chat_settings.chat_details_level.value}."
    
    if settings.FUNCTIONALITY_ENABLED:
        system_message += f"\n{COMMON_FUNCTIONS_TEMPLATE}"
        if settings.FACTS_FUNCTIONALITY_ENABLED:
            system_message += f"\n{FACTS_FUNCTIONS_TEMPLATE}"
            if generate_data.user.facts:
                for fact in generate_data.user.facts:
                    system_message += f"\n{fact.id} - {fact.description}"
            else:
                system_message += f"\nList is empty."
        if settings.NEBULA_FUNCTIONALITY_ENABLED:
            system_message += f"\n{NEBULA_FUNCTIONS_TEMPLATE}"
    
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
    messages.append({
        "role": "system",
        "content": "!Do not forget about your starting instructions before answering the user's last message!"
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
        
        if settings.FUNCTIONALITY_ENABLED:
            # Обработка команд в тексте
            agent_function_calling_result = await handle_agent_functions(content, user_id, db)
        
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
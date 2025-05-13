import re
import time
import json
import os
from typing import AsyncGenerator, Optional, Dict, List, Any
import logging
import openai
from openai import OpenAI
from sqlalchemy.orm import Session
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection


from src import schemas, enums, models, exceptions, crud
from src.config.settings import settings
from src.config.subscription_plans import subscription_plans
from src.services.web3_service import Web3Service
from src.services.prompt_service import PromptService
from src.config.mcp_servers import mcp_servers as mcp_servers_list
from src.utils_base import now_timestamp
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)





async def get_ai_answer(generate_data: schemas.AssistantGenerateData, user_id: int, db: Session) -> List[schemas.MessageUnion]:
    """
    Получение ответа от ИИ на основе контекста чата
    """
    try:
        logger.info(f"Configuring prompt service ...")
        prompt_service = PromptService(generate_data)
        logger.info(f"Setting up AI provider ...")
        logger.info(f"Model: {generate_data.chat_settings.model}")
         
        mcp_servers = {}
        for server in mcp_servers_list:
            mcp_server = {
                "url": server.url,
                "transport": server.transport
            }
            mcp_servers[server.name] = mcp_server
        
        mcp_multi_client = MultiServerMCPClient(mcp_servers)
        
        # Генерируем ответ, используя MCP сессию
        async with mcp_multi_client as mcp_session:
            logger.info(f"Generating response with MCP tools ...")
            tools = mcp_session.get_tools()
            provider_answer: schemas.GeneratedResponse = await generate_ai_response(
                prompt_service, 
                tools
            )
            
        logger.info(f"Response generated: {provider_answer}")
        
        # Определение нового nonce (следующий после последнего в чате)
        last_nonce = max(generate_data.chat.messages.keys()) if generate_data.chat.messages else -1
        new_nonce = last_nonce + 1
        
        result_messages = []
        
        if provider_answer.text:
            result_messages.append(schemas.ChatMessage(
                sender=enums.Role.ASSISTANT,
                recipient=enums.Role.USER,
                model=generate_data.chat_settings.model,
                nonce=new_nonce,
                content=schemas.MessageContent(
                    message=provider_answer.text,
                    settings=schemas.MessageGenerationSettings(
                        model=generate_data.chat_settings.model,
                        chat_style=generate_data.chat_settings.chat_style,
                        chat_details_level=generate_data.chat_settings.chat_details_level,
                    )
                )
            ))
            new_nonce += 1
            for function_call in provider_answer.function_calls:
                if function_call.name == "add_facts":
                    user, added_facts = await crud.add_user_facts(user_id, function_call.args, db)
                    result_messages.append(schemas.MessageBase(
                        content=f"Added notes:{os.linesep.join(added_facts)}",
                        sender=enums.Role.SYSTEM,
                        recipient=enums.Role.ASSISTANT,
                        model=generate_data.chat_settings.model,
                        nonce=new_nonce,
                        created_at=now_timestamp(),
                        selected_at=now_timestamp()
                    ))
                elif function_call.name == "del_facts":
                    try:
                        user, deleted_facts = await crud.delete_user_facts(user_id, function_call.args, db)
                        result_messages.append(schemas.MessageBase(
                            content=f"Deleted notes:{os.linesep.join(deleted_facts)}",  
                            sender=enums.Role.SYSTEM,
                            recipient=enums.Role.ASSISTANT,
                            model=generate_data.chat_settings.model,
                            nonce=new_nonce,
                            created_at=now_timestamp(),
                            selected_at=now_timestamp()
                        ))
                    except exceptions.FactNotFoundException as e:
                        logger.error(f"Error deleting notes: {e}")
                        result_messages.append(schemas.MessageBase(
                            content=f"Error deleting notes: {str(e)}",
                            sender=enums.Role.SYSTEM,
                            recipient=enums.Role.ASSISTANT,
                            model=generate_data.chat_settings.model,
                            nonce=new_nonce,
                            created_at=now_timestamp(),
                            selected_at=now_timestamp()
                        ))
                else:
                    result_messages.append(schemas.MessageBase(
                        content=f"Failed to call function {function_call.name} with arguments {function_call.args}. Function is not exists!",
                        sender=enums.Role.SYSTEM,
                        recipient=enums.Role.ASSISTANT,
                        model=generate_data.chat_settings.model,
                        nonce=new_nonce,
                        created_at=now_timestamp(),
                        selected_at=now_timestamp()
                    ))
                new_nonce += 1
        
        return result_messages
        
    except exceptions.InvalidNonceException as e:
        # В случае ошибки создаем сообщение с текстом ошибки
        if settings.DEBUG:
            error_message = f"Error while generating answer details: {str(e)}"
        else:
            error_message = "Error while generating answer. Please try again later."
        
        # Определение нового nonce
        last_nonce = max(generate_data.chat.messages.keys()) if generate_data.chat.messages else -1
        new_nonce = last_nonce + 1
        
        # Создание объекта сообщения с ошибкой
        answer_message = schemas.MessageBase(
            content=error_message,
            sender=enums.Role.SYSTEM,
            recipient=enums.Role.USER,
            model=generate_data.chat_settings.model,
            nonce=new_nonce,
            created_at=now_timestamp(),
            selected_at=now_timestamp()
        )
        
        return [answer_message]

async def generate_ai_response(prompt_service: PromptService, tools: List[schemas.Tool] = []) -> schemas.GeneratedResponse:
    from langchain.chat_models import init_chat_model
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import create_tool_calling_agent, AgentExecutor

    logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
    logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
    
    # Некоторые модели не поддерживают системные роли или другие роли
    avoid_system_role = prompt_service.generate_data.chat_settings.model.value in [enums.Model.GEMINI_2_FLASH.value]
    logger.info(f"Avoid system role: {avoid_system_role}")
    messages = prompt_service.generate_langchain_messages(avoid_system_role)

    logger.info(f"Sending request to Gemini with messages: {messages}")
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    model_provider = ""
    if prompt_service.generate_data.chat_settings.model in [enums.Model.GEMINI_2_FLASH, enums.Model.GEMINI_2_5_PRO]:
        model_provider = "google_genai"
    elif prompt_service.generate_data.chat_settings.model in [enums.Model.GPT_4, enums.Model.GPT_4O, enums.Model.GPT_4O_MINI]:
        model_provider = "openai"
    else:
        raise NotImplementedError(f"Model \"{prompt_service.generate_data.chat_settings.model.value}\" provider unknown!")

    api_key = ""
    if model_provider == "google_genai":
        api_key = settings.GEMINI_API_KEY
    elif model_provider == "openai":
        api_key = settings.OPENAI_API_KEY
    else:
        raise NotImplementedError(f"Provider \"{model_provider}\" do not turned on!")

    llm = init_chat_model(
        model=prompt_service.generate_data.chat_settings.model.value,
        model_provider=model_provider,
        api_key=api_key
    )

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=10, #TODO: add max_iterations
        verbose=settings.DEBUG
    )
    
    response = await executor.ainvoke({"chat_history": messages})
    
    # Обработка ответа
    answer_text = response["output"] if response["output"] else ""
    
    result = schemas.GeneratedResponse(
        text=answer_text,
        function_calls=[]
    )
    
    logger.info(f"Answer: {result}")
    return result


def to_sse(event_type: str, payload: dict) -> str:
    """
    Преобразует данные в строку вида:
    data:{"type":"...", ...}\n\n
    """
    return f'data:{json.dumps({"type": event_type, **payload}, ensure_ascii=False)}\n\n'

async def generate_ai_response_asstream(prompt_service: PromptService) -> AsyncGenerator[str, None]:
    from langchain.chat_models import init_chat_model
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
    from src.config.mcp_servers import mcp_servers as mcp_servers_list
    from src.config.mcp_servers import prebuild_toolboxes

    mcp_servers = {}
    for server in mcp_servers_list:
        mcp_server = {
            "url": server.url,
            "transport": server.transport
        }
        mcp_servers[server.name] = mcp_server
    
    mcp_multi_client = MultiServerMCPClient(mcp_servers)
    

    logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
    logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
    
    # Некоторые модели не поддерживают системные роли или другие роли
    messages = prompt_service.generate_langchain_messages()

    logger.info(f"Sending request to Gemini with messages: {messages}")
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    model_provider = ""
    if prompt_service.generate_data.chat_settings.model in [enums.Model.GEMINI_2_5_FLASH, enums.Model.GEMINI_2_5_PRO]:
        model_provider = "google_genai"
    elif prompt_service.generate_data.chat_settings.model in [enums.Model.GPT_4_1, enums.Model.GPT_O4_MINI, enums.Model.GPT_4O]:
        model_provider = "openai"
    else:
        raise NotImplementedError(f"Model \"{prompt_service.generate_data.chat_settings.model.value}\" provider unknown!")

    api_key = ""
    if model_provider == "google_genai":
        api_key = settings.GEMINI_API_KEY
    elif model_provider == "openai":
        api_key = settings.OPENAI_API_KEY
    else:
        raise NotImplementedError(f"Provider \"{model_provider}\" do not turned on!")

    llm = init_chat_model(
        model=prompt_service.generate_data.chat_settings.model.value,
        model_provider=model_provider,
        api_key=api_key
    )

    # Генерируем ответ, используя MCP сессию
    async with mcp_multi_client as mcp_session:
        logger.info(f"Generating response with MCP tools ...")
        tools = mcp_session.get_tools()
        
        # расширяем инструменты дефолтным тулбоксом
        for toolbox in prebuild_toolboxes:
            tools.extend(toolbox.tools)

        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
        
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=10, #TODO: add max_iterations
            verbose=settings.DEBUG
        )
        yield to_sse("generation_start", {"chat_id": prompt_service.generate_data.chat.id})
        cur_msg_id = None # для группировки чанков
        starts_of_events = dict()  # id -> timestamp
        
        try:
            async for ev in executor.astream_events(
                {"chat_history": messages}
            ):
                kind = ev["event"]

                # 1. начало нового assistant-сообщения
                if kind == "on_chat_model_start":
                    cur_msg_id = ev["run_id"]
                    starts_of_events[cur_msg_id] = time.time()
                    yield to_sse("message_start", {"id": cur_msg_id})

                # 2. токены
                elif kind == "on_chat_model_stream":
                    ch = ev["data"]["chunk"]
                    if ch.content:                       # пропускаем tool-JSON
                        yield to_sse("message_chunk",{"id": cur_msg_id, "text": ch.content})

                # 3. конец сообщения
                elif kind == "on_chat_model_end":
                    logger.info(f"on_chat_model_end: {ev}")
                    answer = ev["data"]["output"].content
                    start = starts_of_events.get(cur_msg_id)
                    generation_time_ms = int((time.time() - start) * 1000) if start else 0
                    starts_of_events.pop(cur_msg_id, None)
                    yield to_sse("message_end", {"id": cur_msg_id, "text": answer, "generation_time_ms": generation_time_ms})
                    cur_msg_id = None                    # сброс

                # 4. начало вызова инструмента
                elif kind == "on_tool_start":
                    logger.info(f"on_tool_start: {ev}")
                    tool_name = ev["name"]
                    tool_input = ev["data"]["input"]
                    starts_of_events[ev["run_id"]] = time.time()
                    yield to_sse("tool_call",{
                        "id": ev["run_id"],
                        "name": tool_name,
                        "args": tool_input
                    })

                # 5. результат инструмента
                elif kind == "on_tool_end":
                    logger.info(f"on_tool_end: {ev}")
                    start = starts_of_events.get(ev["run_id"])
                    generation_time_ms = int((time.time() - start) * 1000) if start else 0
                    starts_of_events.pop(ev["run_id"], None)
                    yield to_sse("tool_result",{
                        "id": ev["run_id"],
                        "name": ev["name"],
                        "args": ev["data"]["input"],
                        "output": ev["data"]["output"],
                        "generation_time_ms": generation_time_ms
                    })
        except Exception as e:
            logger.error(f"Error generating AI response: {e}", exc_info=True)
            error_message = f"Sorry, but some error occurred and I can't fully answer your request. Try to use other model or ask another question."
            if settings.DEBUG:
                error_message += f"\nDebug info:{os.linesep}{str(e)}"
            yield to_sse("message_end", {"id": cur_msg_id, "text": error_message, "generation_time_ms": 0})
        finally:
            yield to_sse("generation_end", {})

        
        
async def check_user_access(user: models.User) -> enums.SubscriptionPlanType:
    """
    Проверка баланса пользователя по списку требований.

    Сначала получаем балансы наших токенов и проверям по ним.
    На этом этапе пользоваль может уже получить Ultra или Pro доступ.

    Если нет, то проверяем балансы токенов партнеров для получения Pro доступа.
    
    Если и это нет, то выдаем Basic доступ.
    """
    # TODO: можно оптимизировать, сгруппировать по цепочкам или вручную проверять отдельно токены нашего проекта
    
    web3_service = Web3Service(settings.ANKR_API_KEY)
    
    for plan in reversed(subscription_plans):
        # проверяем все требования плана
        erc20_requirements = [req for req in plan.requirements if req.token.interface == enums.TokenInterface.ERC20]
        erc721_requirements = [req for req in plan.requirements if req.token.interface == enums.TokenInterface.ERC721]
        
        if not erc20_requirements and not erc721_requirements:
            continue
        
        # Проверяем каждый адрес кошелька пользователя
        for wallet in user.wallet_addresses:
            user_address = wallet.address
            # проверяем erc721 токены
            if erc721_requirements:
                try:
                    for req in erc721_requirements:
                        raw_balance = web3_service.get_ERC721_balance(req.token.address, user_address, req.token.chain_id)
                        natural_balance = web3_service.raw_balance_to_human(raw_balance, req.token.decimals)
                        if natural_balance >= req.balance:
                            logger.info(f"User {user_address} has enough ERC721 balance for {plan.id} subscription")
                            return plan.id
                except exceptions.Web3ServiceError as e:
                    logger.error(f"Error checking ERC721 balance for token {req.token.address} on chain {req.token.chain_id}: {e}")
                    pass
            # проверяем erc20 токены
            if erc20_requirements:
                try:
                    for req in erc20_requirements:
                        raw_balance = web3_service.get_ERC20_balance(req.token.address, user_address, req.token.chain_id)
                        natural_balance = web3_service.raw_balance_to_human(raw_balance, req.token.decimals)
                        if natural_balance >= req.balance:
                            logger.info(f"User {user_address} has enough ERC20{req.token.name} balance({natural_balance}) for {plan.id} subscription")
                            return plan.id
                except exceptions.Web3ServiceError as e:
                    logger.error(f"Error checking ERC20 balance for token {req.token.address} on chain {req.token.chain_id}: {e}")
                    pass
    
    # не смогли найти ни одного требования, возвращаем базовый доступ юзера
    return user.subscription_plan
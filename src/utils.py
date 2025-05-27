import functools
import re
import time
import json
import os
from typing import AsyncGenerator, Callable, Optional, Dict, List, Any, TypeVar
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
from src.utils_base import now_timestamp, parse_date_to_timestamp
import inspect
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# декоратор для отлова ошибок и возвращения json
def catch_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Возвращает обёртку-клон. При исключении → {"error": ...}."""
    debug = getattr(settings, "DEBUG", False)

    if inspect.iscoroutinefunction(fn):              # async-функция / coroutine
        @functools.wraps(fn)
        async def _async(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                return json.dumps({"error": str(e) if debug else "Error occurred while executing tool."})
        return _async                                # type: ignore[return-value]

    @functools.wraps(fn)                             # обычная sync-функция
    def _sync(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return json.dumps({"error": str(e) if debug else "Error occurred while executing tool."})
    return _sync  

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
        
        # Генерируем ответ, используя MCP инструменты
        logger.info(f"Generating response with MCP tools ...")
        tools = await mcp_multi_client.get_tools()
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
                    except exceptions.FactNotFoundError as e:
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
        
    except exceptions.InvalidNonceError as e:
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

    logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
    logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
    
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

    mcp_servers = {}
    for toolbox_id, server in mcp_servers_list.items():
        mcp_server = {
            "url": server.url,
            "transport": server.transport
        }
        if toolbox_id in prompt_service.generate_data.toolbox_ids:
            mcp_servers[server.name] = mcp_server
    
    mcp_multi_client = MultiServerMCPClient(mcp_servers)
    
    # Генерируем ответ, используя MCP инструменты
    logger.info(f"Generating response with MCP tools ...")
    tools = await mcp_multi_client.get_tools()
    
    # расширяем инструменты дефолтным тулбоксом
    for toolbox in prebuild_toolboxes:
        if toolbox.id in prompt_service.generate_data.toolbox_ids:
            tools.extend(toolbox.tools)

    # декорируем инструменты чтобы отлавливать ошибки
    for tool in tools:
        if hasattr(tool, "func") and callable(tool.func):
            tool.func = catch_errors(tool.func)          # sync-инструмент
        elif hasattr(tool, "coroutine") and callable(tool.coroutine):
            tool.coroutine = catch_errors(tool.coroutine)  # async-инструмент
        else:
            logger.error(f"Error: Tool {tool.name} has no func or coroutine attribute")

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
        yield to_sse("message_start", {"id": cur_msg_id})
        yield to_sse("message_chunk", {"id": cur_msg_id, "text": error_message})
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

from src.services.x_api_service import *


async def update_project_data(project: models.Project, from_timestamp: int, db: Session) -> bool:
    """
    Обновление данных проекта.
    
    Находит новые твиты и парсит их, чтобы отразить в базе данных.
    
    from_timestamp - timestamp в секундах, с которого нужно искать твиты. Это либо время создания проекта, либо максимальный срок, который мы учитываем в статистике.
    """
    # сервис для работы с x api
    x_api_service = XApiService()
    query = f'{project.name} since:{from_timestamp}'
    
    try:
        # Получаем твитов сколько сможем
        response: XApiService.FeedResponse = await x_api_service.search(query, from_timestamp)
        
        if not response or not response.tweets:
            logger.info(f"No tweets found for project {project.name} with query {query}")
            return True
            
        logger.info(f"Found {len(response.tweets)} tweets for project {project.name}")
        
        for tweet in response.tweets:
            try:
                # Проверяем наличие необходимых полей
                if not hasattr(tweet, 'legacy') or not hasattr(tweet, 'rest_id'):
                    logger.warning(f"Skipping tweet due to missing required fields: {tweet}")
                    continue

                main_post_data = tweet.legacy
                # Получаем количество просмотров, если они есть, иначе 0
                views = 0
                if hasattr(tweet, 'views') and tweet.views and hasattr(tweet.views, 'count') and tweet.views.count is not None:
                    try:
                        views = int(tweet.views.count)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert views count to integer: {tweet.views.count}")
                        views = 0
                
                post_id = tweet.rest_id
                post_db: models.SocialPost | None = await crud.get_social_post_by_id(post_id, db)
                
                if not post_db:
                    # Проверяем наличие данных об источнике
                    if not hasattr(tweet, 'core') or not hasattr(tweet.core, 'user_results') or not hasattr(tweet.core.user_results, 'result'):
                        logger.warning(f"Skipping tweet due to missing source data: {tweet}")
                        continue

                    source_id = tweet.core.user_results.result.rest_id
                    source_db: models.SocialAccount | None = await crud.get_social_account_by_id(source_id, db)
                    if not source_db:
                        # создаем источник
                        source_data: XApiService.User = tweet.core.user_results.result
                        if not hasattr(source_data, 'legacy'):
                            logger.warning(f"Skipping tweet due to missing source legacy data: {source_data}")
                            continue

                        source_db = models.SocialAccount(
                            social_id=source_id,
                            name=source_data.legacy.name,
                            social_login=source_data.legacy.screen_name,
                        )
                        source_db = await crud.create_social_account(source_db, db)

                    # Преобразуем строку даты в timestamp
                    posted_at_timestamp = parse_date_to_timestamp(main_post_data.created_at)
                    
                    # создаем пост
                    post_db = models.SocialPost(
                        social_id=post_id,
                        account_id=source_db.id,
                        text=main_post_data.full_text,
                        posted_at=posted_at_timestamp,
                        raw_data=tweet.model_dump(mode="json")
                    )
                    post_db = await crud.create_social_post(post_db, db)

                # создаем связь между постом и проектом, если ее еще не существует
                await crud.create_or_deny_project_post_mention(project.id, post_db.id, db)
                
                # сохраняем статистику поста
                post_stats_db = models.SocialPostStatistic(
                    post_id=post_db.id,
                    likes=main_post_data.favorite_count,
                    comments=main_post_data.reply_count,
                    reposts=main_post_data.retweet_count,
                    views=views
                )
                await crud.create_social_post_statistic(post_stats_db, db)
            except Exception as e:
                logger.error(f"Error processing tweet: {str(e)}")
                logger.error(f"Tweet data: {tweet}")
                continue

    except Exception as e:
        logger.error(f"Error fetching tweets: {str(e)}")
        logger.error(f"Project: {project.name}, Query: {query}")
        return False

    return True



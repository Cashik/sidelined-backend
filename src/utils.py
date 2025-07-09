from datetime import datetime
from enum import Enum
import functools
import random
import re
import time
import json
import os
import traceback
from typing import AsyncGenerator, Callable, Optional, Dict, List, Any, TypeVar
import logging
import openai
from openai import OpenAI
from sqlalchemy.orm import Session, aliased, joinedload
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
import requests
import inspect
from collections import defaultdict

from src import schemas, enums, models, exceptions, crud, utils_base
from src.config.settings import settings
from src.config.subscription_plans import subscription_plans
from src.config.ai_models import all_ai_models as ai_models
from src.services.web3_service import Web3Service
from src.services.prompt_service import PromptService
from src.services import cache_service
from src.services import x_api_service
from src.config.mcp_servers import mcp_servers as mcp_servers_list
from src.config import leaderboard
from src.utils_base import now_timestamp, parse_date_to_timestamp

from src.services.cache_service import LeaderboardCacheService, FeedCacheService
from src.crud import get_top_engagement_posts

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

# TODO: не используется
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

# TODO: не используется
async def generate_ai_response(prompt_service: PromptService, tools: List[schemas.Tool] = []) -> schemas.GeneratedResponse:
    from langchain.chat_models import init_chat_model
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents import create_tool_calling_agent, AgentExecutor

    logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
    logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
    
    # Некоторые модели не поддерживают системные роли или другие роли
    avoid_system_role = prompt_service.generate_data.chat_settings.model.value in [enums.Model.GEMINI_2_5_FLASH.value]
    logger.info(f"Avoid system role: {avoid_system_role}")
    messages = prompt_service.generate_langchain_messages(avoid_system_role)

    logger.info(f"Sending request to Gemini with messages: {messages}")
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    model_provider = ""
    if prompt_service.generate_data.chat_settings.model in [enums.Model.GEMINI_2_5_FLASH, enums.Model.GEMINI_2_5_PRO]:
        model_provider = "google_genai"
    elif prompt_service.generate_data.chat_settings.model in [enums.Model.GPT_4_1, enums.Model.GPT_4O, enums.Model.GPT_O4_MINI]:
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
    import time
    import os
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    from langchain_core.messages import ToolMessage

    # --- Сбор тулзов MCP ---
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from src.config.mcp_servers import mcp_servers as mcp_servers_list
    from src.config.mcp_servers import prebuild_toolboxes
    from src import utils_base

    start_ts = utils_base.now_timestamp()

    logger.info(f"[Generation]: Generating response for prompt: {prompt_service.generate_data.chat.messages}")
    logger.info(f"[Generation]: Model: {prompt_service.generate_data.chat_settings.model}")
    
    messages = prompt_service.generate_langchain_messages()

    
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
            logger.error(f"Error: Tool {tool} has no func or coroutine attribute")

    # Инициализируем модель
    model_provider = ""
    if prompt_service.generate_data.chat_settings.model in [enums.Model.GEMINI_2_5_FLASH, enums.Model.GEMINI_2_5_PRO]:
        model_provider = "google_genai"
    elif prompt_service.generate_data.chat_settings.model in [enums.Model.GPT_4_1, enums.Model.GPT_O4_MINI, enums.Model.GPT_4O]:
        model_provider = "openai"
    else:
        raise NotImplementedError(f"Model \"{prompt_service.generate_data.chat_settings.model.value}\" provider unknown!")

    import os
    if model_provider == "google_genai":
        os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY
    elif model_provider == "openai":
        # Добавляем тулз для поиска в интернете
        tools.append({"type": "web_search_preview"})
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    else:
        raise NotImplementedError(f"Provider \"{model_provider}\" do not turned on!")

    logger.info(f"[Generation]:{utils_base.now_timestamp() - start_ts} Using model: {model_provider}:{prompt_service.generate_data.chat_settings.model.value}")

    model = f"{model_provider}:{prompt_service.generate_data.chat_settings.model.value}"

    # --- Создаём агента через LangGraph ---
    agent = create_react_agent(
        model=model,
        tools=tools
        # prompt/prompt_template если хочешь переопределять, но в большинстве случаев дефолт ок!
    )
    logger.info(f"[Generation]: Agent created: {agent}")
    #test_response = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]})
    #logger.info(f"Test response: {test_response}")

    yield to_sse("generation_start", {"chat_id": prompt_service.generate_data.chat.id})
    cur_msg_id = None # для группировки чанков
    starts_of_events = dict()  # id -> timestamp
    logger.info(f"[Generation]:{utils_base.now_timestamp() - start_ts} Stream started")
    try:
        async for ev in agent.astream_events({"messages": messages}):
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
                    yield to_sse("message_chunk",{"id": cur_msg_id, "text": ch.text()})

            # 3. конец сообщения
            elif kind == "on_chat_model_end":
                logger.info(f"on_chat_model_end: {ev}")
                answer = ev["data"]["output"].text()
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

                output = ev["data"]["output"]
                if isinstance(output, ToolMessage):
                    output = output.content

                yield to_sse("tool_result",{
                    "id": ev["run_id"],
                    "name": ev["name"],
                    "args": ev["data"]["input"],
                    "output": output,
                    "generation_time_ms": generation_time_ms
                })
            else:
                #logger.info(f"Unknown event: {ev}")
                pass
                
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
            logger.info(f"Checking wallet {user_address} for plan {plan.id} chain_family={wallet.chain_family}")
            if wallet.chain_family == enums.ChainFamily.SOLANA:
                #TODO: проверка токенов для solana
                logger.warn(f"Skip token validation for solana user_address={user_address} wallet_id={wallet.id}")
            elif wallet.chain_family == enums.ChainFamily.EVM:
                logger.info(f"Checking EVM wallet {user_address} for plan {plan.id}")
                # проверяем erc721 токены
                if erc721_requirements:
                    try:
                        for req in erc721_requirements:
                            raw_balance = web3_service.get_ERC721_balance(req.token.address, user_address, req.token.chain_id)
                            natural_balance = web3_service.raw_balance_to_human(raw_balance, req.token.decimals)
                            if natural_balance >= req.balance:
                                logger.info(f"User {user_address} has enough ERC721 balance for {plan.id} subscription")
                                return plan.id
                            logger.info(f"User {user_address} has no enough ERC721 balance for {plan.id} subscription, required: {req.balance}, actual: {natural_balance}")
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
                            logger.info(f"User {user_address} has no enough ERC20{req.token.name} balance({natural_balance}) for {plan.id} subscription, required: {req.balance}, actual: {natural_balance}")
                    except exceptions.Web3ServiceError as e:
                        logger.error(f"Error checking ERC20 balance for token {req.token.address} on chain {req.token.chain_id}: {e}")
                        pass
    
    # не смогли найти ни одного требования, возвращаем базовый доступ юзера
    return user.subscription_plan

async def check_user_subscription(user: models.User, db: Session) -> enums.SubscriptionPlanType:
    """
    Проверка подписки пользователя.
    """
    logger.info(f"check_user_subscription: {user.id}")
    user_plan_check: models.UserPlanCheck | None = crud.get_user_last_plan_check(db, user.id)
    logger.info(f"check_user_subscription: {user_plan_check}")
    if user_plan_check:
        logger.info(f"check_user_subscription: {user_plan_check.created_at + settings.BALANCE_CHECK_LIFETIME_SECONDS} > {utils_base.now_timestamp()}")
        if user_plan_check.created_at + settings.BALANCE_CHECK_LIFETIME_SECONDS > utils_base.now_timestamp():
            logger.info(f"check_user_subscription: {user_plan_check.user_plan}")
            return user_plan_check.user_plan
    logger.info(f"check_user_subscription: no valid check")
    # Нет валидной проверки, создаем новую
    current_plan: enums.SubscriptionPlanType = await check_user_access(user)
    # сохраняем проверку
    crud.create_user_plan_check(db, user.id, current_plan)
    # возвращаем текущую подписку
    return current_plan

from src.services.x_api_service import *


async def update_project_feed(project: models.Project, from_timestamp: int, db: Session) -> bool:
    """
    Обновление фида проекта.
    Тут нам важно, чтобы в твите было упоминание проекта и твит имел достаточное количество лайков.
    """
    keywords = project.keywords.split(";")
    keywords = [f'"{kw}"' for kw in keywords if kw]
    keywords_query = " OR ".join(keywords)
    min_likes = project.search_min_likes if project.search_min_likes is not None else settings.DEFAULT_MINIMAL_LIKES_TO_SEARCH_TWEETS
    query = f"({keywords_query}) min_faves:{min_likes}"
    return await _update_posts_data(project.id, query, from_timestamp, db)

async def update_project_news(project: models.Project, from_timestamp: int, db: Session) -> bool:
    """
    Обновление новостей проекта.
    Тут нам важно только чтобы твит был от связанного аккаунта проекта.
    """
    logins = []
    for account in project.accounts:
        logins.append(f"from:{account.account.social_login}")
    
    logins_query = " OR ".join(logins)
    if len(logins) == 0:
        logger.info(f"update_project_news: no accounts found for project {project.id}")
        return True
    elif len(logins) == 1:
        query = f"{logins_query}"
    else:
        query = f"({logins_query})"
    
    logger.info(f"update_project_news query: {query}")
    return await _update_posts_data(project.id, query, from_timestamp, db)


async def _update_posts_data(project_id: int, query: str, from_timestamp: int, db: Session) -> bool:
    """
    Обновление данных проекта.
    
    Находит новые твиты и парсит их, чтобы отразить в базе данных.
    
    from_timestamp - timestamp в секундах, с которого нужно искать твиты. Это либо время создания проекта, либо максимальный срок, который мы учитываем в статистике.
    """

    tweets_added_count = 0
    tweets_skipped_count = 0
    
    # добавляем фильтр по типу
    query += f" -filter:replies"
    
    try:
        # сервис для работы с x api
        x_api_service = XApiService()
        # получаем твиты
        response: XApiService.FeedResponse = await x_api_service.search(query, from_timestamp)
        
        if not response or not response.tweets:
            logger.info(f"No tweets found for query {query}")
            return True
    except Exception as e:
        logger.error(f"Error fetching tweets: {str(e)} for query {query}")
        return False
    
    logger.info(f"Found {len(response.tweets)} tweets for query {query}")
    
    for tweet in response.tweets:
        try:
            # Проверяем наличие необходимых полей 
            #     if not hasattr(post, 'legacy') or not hasattr(post, 'rest_id'):
            if not hasattr(tweet, 'legacy') or not hasattr(tweet, 'rest_id'):
                logger.warning(f"Skipping tweet due to missing required fields: {tweet}")
                continue
            
            # пропускаем невалидные посты
            if not _post_is_valid(tweet):
                logger.info(f"Skipping. post: {str(tweet.legacy.full_text)[:100]} Is retweet: {tweet.is_retweet()} Is quote: {tweet.is_quote()} Is reply: {tweet.is_reply()} Is original: {tweet.is_original()}")
                tweets_skipped_count += 1
                continue
            
            # сохраняем пост и его статистику
            await _update_post_data(tweet, project_id, db)
            tweets_added_count += 1
        except Exception as e:
            logger.error(f"Error processing tweet: {str(e)}")
            logger.error(f"Tweet data: {tweet}")
            continue

    logger.info(f"Tweets added: {tweets_added_count}, skipped: {tweets_skipped_count}")
    return True

def _post_is_valid(post: x_api_service.TweetResult) -> bool:
    """
    Проверка поста.
    """    
    if post.is_quote() or post.is_original():
        return True
    return False


async def _update_post_data(post: x_api_service.TweetResult, project_id: int, db: Session):
    """
    Обновление данных поста.
    """
    post_id = post.rest_id
    post_db: models.SocialPost | None = await crud.get_social_post_by_id(post_id, db)
    main_post_data = post.legacy
    views = 0
    if hasattr(post, 'views') and post.views and hasattr(post.views, 'count') and post.views.count is not None:
        try:
            views = int(post.views.count)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert views count to integer: {post.views.count}")
            views = 0
    if not post_db:
        # Проверяем наличие данных об источнике
        if not hasattr(post, 'core') or not hasattr(post.core, 'user_results') or not hasattr(post.core.user_results, 'result'):
            logger.warning(f"Skipping tweet due to missing source data: {post}")
            return
        
        source_data: XApiService.User = post.core.user_results.result

        # доп фильтрация
        # если (like+comments+retweet)x3 < views , то пропускаем
        if (main_post_data.favorite_count + main_post_data.reply_count + main_post_data.retweet_count) * 3 > views and views!=0:
            logger.info(f"Skipping tweet due speculative engagement: {3*(main_post_data.favorite_count + main_post_data.reply_count + main_post_data.retweet_count)} > {views}")
            return
        # если текст меньше 30 символов, то пропускаем
        if len(main_post_data.full_text) < 30:
            logger.info(f"Skipping tweet due to short text: {main_post_data.full_text}")
            return
        # если в текте больше 5 упоминаний - пропускаем
        if main_post_data.full_text.count("@") > 5:
            logger.info(f"Skipping tweet due to many mentions: {main_post_data.full_text}")
            return

        source_id = post.core.user_results.result.rest_id
        social_account_db: models.SocialAccount | None = await crud.get_social_account_by_id(source_id, db)
        
        if not social_account_db:
            # создаем источник (аккаунт в твиттере)
            if not hasattr(source_data, 'legacy'):
                logger.warning(f"Skipping tweet due to missing source legacy data: {source_data}")
                return
            social_account_db = models.SocialAccount(
                social_id=source_id,
                name=source_data.legacy.name,
                social_login=source_data.legacy.screen_name,
                last_avatar_url=source_data.legacy.profile_image_url_https,
                last_followers_count=source_data.legacy.followers_count
            )
            social_account_db = await crud.create_or_update_social_account(social_account_db, db)
        else:
            # TODO: обновлять аватар и подписчиков желательно только если проект в лидерборде.
            # обновляем только если есть изменения
            need_update = False
            if source_data.legacy.profile_image_url_https is not None and social_account_db.last_avatar_url != source_data.legacy.profile_image_url_https:
                social_account_db.last_avatar_url = source_data.legacy.profile_image_url_https
                need_update = True
            if source_data.legacy.followers_count is not None and social_account_db.last_followers_count != source_data.legacy.followers_count:
                social_account_db.last_followers_count = source_data.legacy.followers_count
                need_update = True
            if need_update:
                social_account_db = await crud.create_or_update_social_account(social_account_db, db)
        
        
        # Преобразуем строку даты в timestamp
        posted_at_timestamp = parse_date_to_timestamp(main_post_data.created_at)
        
        # создаем пост
        post_db = models.SocialPost(
            social_id=post_id,
            account_id=social_account_db.id,
            text=main_post_data.full_text,
            posted_at=posted_at_timestamp,
            raw_data=post.model_dump(mode="json")
        )
        post_db = await crud.create_social_post(post_db, db)

    # создаем связь между постом и проектом, если ее еще не существует
    await crud.create_or_deny_project_post_mention(project_id, post_db.id, db)
    
    # сохраняем статистику поста
    post_stats_db = models.SocialPostStatistic(
        post_id=post_db.id,
        likes=main_post_data.favorite_count,
        comments=main_post_data.reply_count,
        reposts=main_post_data.retweet_count,
        views=views
    )
    await crud.create_social_post_statistic(post_stats_db, db)
    return




def calculate_post_engagement_score_for_shema(post: schemas.Post) -> float:
    """
    Расчет показателя вовлеченности поста.
    """
    return calculate_post_engagement_score(post.stats.views_count, post.stats.favorite_count, post.stats.retweet_count, post.stats.reply_count)

def calculate_post_engagement_score(views: int, likes: int, reposts: int, comments: int) -> float:
    """
    Расчет показателя вовлеченности поста.
    """
    return views*0.001 + likes * 1 + reposts * 3 + comments * 4

def convert_posts_to_schemas(posts: List[models.SocialPost]) -> List[schemas.Post]:
    """
    Преобразование списка моделей постов в список схем.
    """
    result = []
    for post in posts:
        try:
            # Берём самую свежую статистику (по created_at)
            stats = max(post.statistic, key=lambda s: s.created_at) if post.statistic else None
            
            # Получаем статусы аккаунта в проектах
            account_projects_statuses = []
            for project_status in post.account.projects:
                account_projects_statuses.append(schemas.SourceStatusInProject(
                    project_id=project_status.project_id,
                    source_type=project_status.type
                ))
            
            # Создаем схему поста
            post_schema = schemas.Post(
                text=post.text,
                created_timestamp=post.posted_at,
                full_post_json=post.raw_data,  # В данном случае полный текст совпадает с обычным
                stats=schemas.PostStats(
                    favorite_count=stats.likes if stats else 0,
                    retweet_count=stats.reposts if stats else 0,
                    reply_count=stats.comments if stats else 0,
                    views_count=stats.views if stats else 0
                ),
                account_name=post.account.name,
                account_projects_statuses=account_projects_statuses
            )
            result.append(post_schema)
        except Exception as e:
            logger.error(f"Error converting post {post.id} to schema: {str(e)}")
            continue
    return result


async def create_project_autoyaps(project: models.Project, db: Session) -> List[schemas.PostExample]:
    """
    Создание auto-yaps(шаблоны для авто-постов) для пользователя по его выбранным проектам и настройкам.
    
    Шаблоны создаются для каждого выделенного пользователем проекта отдельно.
    """
    # TODO: проверяем, если ли у пользователя возможность создавать авто-посты (лимит или подписка)
    # TODO: нужно ли забирать кредиты у пользователя за создание авто-постов?
    feed_limit = 100
    feed_timestamp_period = 60 * 60 * 24 * 1  # 1 day
    templates = []
    
    default_model = ai_models[0]
    # получаем фид проекта
    # TODO: возможно, стоит кешировать фид проекта
    project_posts: List[models.SocialPost] = await crud.get_project_feed(project, feed_timestamp_period, db)
    
    # преобразуем в схему
    project_feed = convert_posts_to_schemas(project_posts)
    # сортируем по убыванию популярности
    project_feed.sort(key=lambda x: calculate_post_engagement_score_for_shema(x), reverse=True)
    # убираем самые непопулярные посты
    project_feed = project_feed[:feed_limit]
    # сортируем еще раз по времени публикации 
    project_feed.sort(key=lambda x: x.created_timestamp, reverse=False)
    
    generation_settings: schemas.AutoYapsGenerationSettings = schemas.AutoYapsGenerationSettings(
        project_feed=project_feed,
        model=default_model,
        project_name=project.name
    )

    examples_posts: List[str] = await generate_pots_examples(generation_settings)
    for post in examples_posts:
        templates.append(schemas.PostExampleCreate(
            project_id=project.id,
            post_text=post
        ))
    
    # сохраняем шаблоны
    result = await crud.create_post_examples(templates, db)
    return result


async def generate_pots_examples(generation_settings: schemas.AutoYapsGenerationSettings, count: int = 5) -> List[str]:
    """
    Генерация примеров постов (твитов) на основе переданных настроек.

    Использует native structured-output LangChain (`with_structured_output`),
    поэтому модель обязана вернуть JSON с ключом `posts`.
    Возвращает список ровно `count` твитов.
    """
    # --- local imports, чтобы не тащить их при старте всего приложения ---
    from langchain.chat_models import init_chat_model
    from pydantic import BaseModel, Field, ValidationError

    # 1. Подготовка входных данных (усечённые тексты, чтобы не раздуть prompt)
    def _prepare_posts(posts: List[schemas.Post], limit: Optional[int] = None, max_len: int = 400) -> str:
        """Concatenates the first `limit` posts, truncating each to `max_len` characters."""
        result = ""
        for i, p in enumerate(posts[:limit if limit else len(posts)]):
            text = p.text[:max(len(p.text), max_len)]
            result += f"Post number {i+1} from {p.account_name}:\n{text}\n\n"
        return result

    project_feed_str = _prepare_posts(generation_settings.project_feed)
    logger.info(f"Project feed: {project_feed_str}")

     #logger.info(f"Generate posts examples prompt: \n{SYSTEM_PROMPT}")
    class Topics(Enum):
        NEWS = "news"
        DISCUSSION = "discussion"
        REFLECTION = "reflection"
        CALL_TO_ACTION = "call-to-action"
        OTHER = "other"
    
    class Tones(Enum):
        SKEPTICAL = "skeptical"
        ENTHUSIASTIC = "enthusiastic"
        REFLECTIVE = "reflective"
        NEUTRAL = "neutral"
        OTHER = "other"
    
    class LengthOption(Enum):
        SHORT = "short. 1-4 sentences"
        MEDIUM = "medium. 1-2 paragraphs, 5-10 sentences, maybe some lists"
        LONG = "long. 2-3 paragraphs, 10-20 sentences, maybe some lists, detailed"
    
    class YesNo(Enum):
        YES = "yes"
        NO = "no"
    
    # 3. Response schema
    class PostSettings(BaseModel):
        topic: Topics = Field(description="Topic of the tweet")
        tone: Tones = Field(description="Tone of the tweet")
        length: LengthOption = Field(description="Length of the tweet. Short: 1-4 sentences, Medium: one long or more than two paragraphs, 4 or more sentences, maybe some lists, Long: 3 or more paragraphs, 5 or more sentences, maybe some lists, detailed")
        emojis: YesNo = Field(description="Use or not use emojis in the tweet")
        hashtags: YesNo = Field(description="Use or not use hashtags in the tweet")
        lists: YesNo = Field(description="Use or not use lists in the tweet")
        question: YesNo = Field(description="Use or not use opening question in the tweet")
        
    class GeneratedPost(BaseModel):
        settings_number: int = Field(description="Number of the template settings that was used to generate the tweet")
        full_text: str = Field(description="Full text of the tweet, strictly adhere to the selected template settings")
        
    class GeneratePostsResponse(BaseModel):
        posts: List[GeneratedPost] = Field(description=f"List of {count} tweets that you came up with taking into account all the rules.")

    examples = []
    examples_str = ""
    for i in range(0, count+3):
        current_template = PostSettings(
            topic=random.choice(list(Topics)),
            tone=random.choice(list(Tones)),
            length=random.choice(list(LengthOption)),
            emojis=random.choice(list(YesNo)),
            hashtags=YesNo.NO,
            lists=random.choice(list(YesNo)),
            question=random.choice(list(YesNo)),
        )
        examples.append(current_template)
        examples_str += f"\nTemplate №{i+1}:{current_template.model_dump_json()}"
        
    # 2. Prompt (TODO: вынести в PromptService)
    SYSTEM_PROMPT = f"""
You are a social media content creator AI agent.
Your task it to analyze some project feed (in X social network) and generate tweets about the project like a human based on one of the given templates.

#Important rules:
- Return the result strictly in JSON format according to the tool schema, without any additional text.
- Analyze the project's feed to highlight all topics or lastests news for a new tweets.
- Write in first or third person, as if you are a regular user.
- You ARE NOT THE PART OF THE PROJECT.
- Do not mention the project directly (the system will do it) and do not mention anyone.
- Tweet should be about main project only {generation_settings.project_name} not about other projects or topics.
- Use english language.

#Content advice:
1. What can you write about?
1.1. current events in the project
1.2. coverage of discussed topics
1.3. wishes, suggestions and thoughts about the project
1.4. forecasts, assessments and other reflections
1.5. other topics that discussed in the project feed
2.What should you not write about?
2.1. answers to tweets of other users
2.2. about the fact that you are an AI agent and your posts are based on the feed
2.3. about the fact that you are using special tools for generating posts
2.4. unfounded statements and false facts

#Tweet variety guidelines:
1. Allways use `\\n` for new lines in this cases:
- for lists
- after opening question
- between paragraphs or not related sentences
- before hashtags
- to separate important information
2. Strict to the selected template settings.


#Examples of good tweets:
##Short:
```
hyperliquid needs a mobile app

they already nailed product and UX on desktop

the moment they go mobile with that same experience, it's the end of cex

—-

Bitcoin is about to break their ATH, TWICE IN THE SAME YEAR, yet my entire TL is sad

Why?

—---

collected some crypto art on @ethereum over the last few months

has been fun to go back and spend time with older collections, incl a lot of code-based work onchain

will keep sharing a few of my new favs 

—----

You can clearly tell that most crypto VCs are mostly beta chasers. 

I've had like 5 different conversations with people from 5 different funds and they're all trying to search for 'The next Hyperliquid' without asking any questions about the underlying dynamics of perps

—-----


who are the strongest emerging founders on @solana right now?

what are they building?


—-
```
##Medium:
```
crypto and free-market libertarianism are actually somewhat uneasy bedfellows

imo part of the power cryptosystems is the way they make us *less* free...by enabling binding commitments, unbreachable contracts....

hyper-free-markets thinking leads to the "efficient breach" theory of contract, where it's good to be able to break your agreements if it's profit-maximizing...at least part of the use-case of crypto is creating agreements you have to stick with even if not profit-maximizing...

overlap between market maxis and crypto maxis is only partial imo, if you really think deeply about it . .

—--------

Nothing PISSES me off more than when bears do a whole personality switch up and start acting bullish.

*Posts pic of BTC tape* — "Hmmm interesting. Looks healthy! Think we have a real shot to go up from here! Things are finally looking promising!"

What?

Didn't you JUST dedicate weeks of your life to bearposting?

What do you mean it "looks good now" when we're about to break ATH you stupid fuck?


—-----

100 supply Overworld Keys went to 18e in the worst bear market for NFTs for a game that literally never existed and a company that didn't deliver *anything*

My gut is Ethos Validators are still undervalued. Hardest thing about the first mover in a new primitive or vertical is pricing because there are no useful/valid comps.

Leads to extreme price inefficiency in practice most of the time.

ico still has his airdropped validator

—------

just deposited some $$ in @katana to scoop some $KAT 

- katana is a defi zkEVM chain incubated by polygon labs & gsr 
- katana uses conduit to batch and roll up txs off-chain, then publishes succinct zk proofs on-chain 
- pre-deposit phase allows you to mint krates (loot-boxes) 
- you park eth, usdc, usdt, or wbtc and get krates 
- the assets you deposit go straight to yearn v3 vaults for baseline yield (these vaults optimise strategies across various defi strategies across morpho, sushi, vertex) 
- vaultbridge re-funnels bridged tvl & network fees back into those same vaults  
- they've just launched a leaderboard on kaito (10m KAT will be distributed as the total prize pool for yappers)

—----

My friend wants to buy a multifamily house

He thinks it will be a good investment, a good business

I disagree completely

After going through my own experience with property management - I am now 1000% all in on the

"Bitcoin is better than real estate in every way" philosophy

```
##Long:
```
HODL is dead.

My biggest mistake this cycle was blindly holding ETH.

Even when my No 1. bullish case for $ETH failed to materialize - super high yields on ETH thanks to Restaking

I love crypto and trading due to its brutal honesty: if you're right = profit, wrong = loss.

But I got complacent with $ETH as narratives shift fast:

- "Ultrasound money" & ETH's ESG edge... Gone.

- Restaking superyields didn't materialize.

- ETH modular vs SOL's monolithic narrative: SOL proved stronger at least short run.

I ignored changing realities, sticking to HODL instead of adapting.

This cycle was more difficult than the last and crypto moves too quickly for passive holds (maybe except BTC).

Profits now come from more active trading, not hoping assets rebound "someday."

$ETH terrible under-performance was a wake up call.

Thankfully rotating to $HYPE saved me.

Still, $ETH has a place in everyone's portfolio and EF is not complacent anymore.

But emerging "Stablecoin & RWAs" L1s with gas abstraction is another growing risk for ETH.

Happy $HYPE ATH for those who celebrate.


—----------------------------


Habibis, listen up

Big news from @peaq that's been quietly in the works for a while:

They've officially launched the world's first Machine Economy Free Zone (MEFZ) in the UAE - a real-world sandbox to test, deploy, and scale machine economy applications with full regulatory and investment backing.

For those who don't realize how massive this is:

This marks another huge leap forward in peaq's mission to build the universal coordination layer for DePINs, machines, and robots - now grounded in a physical zone with clear regulation, real adoption paths, R&D infrastructure, and investment / liquidity pipelines.

In my opinion, this just made peaq orders of magnitude more relevant for anyone building in DePIN, DePAI, robotics, and machine economy verticals.

Expecting the peaqosystem to experience explosive growth from here - with fresh use cases, stronger network effects, and more real-world traction than ever.

And it'll be all powered by $PEAQ at the core.


—----


In case you missed it: the big money is coming for your $ETH - aggressively.

Yesterday alone saw over $125M in net inflows into spot ETH ETFs, which marks the largest daily inflow in the past 4 months.

But more impressively, ETH ETFs are now on a record-breaking streak of consecutive daily inflows, totaling over $1 billion in 17 days. 

That's not just bullish - that's conviction.

For the first time ever, total assets under management have now exceeded $10 billion.

Macro. Narratives. Fundamentals. Ecosystem. Execution.

Everything is aligning for Ethereum, and smart money is clearly watching.

Wouldn't be surprised if June/July ends up being the last time we ever see ETH under $3K.

—-----

Yesterday we made a post about a small number of OS1 features being depreciated.

I understand that some of you used these regularly & feel passionately about them - especially Deals.

Decisions like these are never made lightly. But there are reasons for them that aren't always obvious. With Deals as an example, the functionality simply doesn't work with ERC721-C contracts where royalties are enforced. As more and more of these contracts have been deployed, the product felt increasingly broken and inconsistently useful.

We might reconsider adding back some of these features in the future. But when you're building at this level of scale and velocity, we sometimes need to make difficult decisions re: prioritization.

I promise you, we have some truly wonderful things in the works.

Please keep the feedback coming. If you love something about OpenSea, be vocal about it. We're listening and will continue to take these things into account as we build out our future roadmap.


—---
main reason not to fade ethos imo is because the market often doesnt know how to price weird/novel tokens

as far as i know this is the only reputation product with some traction and thus has little to no live comps

compare this to gaming tokens or ethereum L2s which comparatively have very clear ceilings from all the fresh tokens introduced in the last 2 years

platform still has obvious issues though

the list of people with the highest reputation is mostly a circlejerk of KOLs who became power users and the recent "listings" feature is a bit of a meme since you can't extract meaningful signal if there's nothing at stake when users vote bullish or bearish

think ethos ends up surprising many as long as the founder doesnt go after  farmers from 3rd world countries again


—---

Many people think that it's bearish that Ethereum is trying to win it all (settlement, execution and SoV).

I believe that Ethereum will win it all and winning all 3 of these prongs is incredibly bullish and why ETH is going to one day be worth $100 trillion+.

Ethereum already won settlement - it's the world's greatest, most decentralised, most secure, most censorship-resistant network to ever exist. Zero downtime. Ultimate reliability.

Ethereum will win execution with its layer 2 ecosystem offering the fastest and cheapest transactions possible while settling on the world's greatest network. The ultimate user experience.

ETH will become the greatest store of value asset to ever exist giving holders the absolute strongest property rights that humanity has ever known. There is no second best.

The world computer is Ethereum and the ticker is ETH.



—-----



distribution is not destiny

look at TON, who is squandering the biggest distribution advantage in crypto history

telegram is the only platform every crypto-native trader and builder lives on. not to mention a billion other retail users

and yet the chain is soulless. no presence. no story. no purpose. 

the success stories of the ecosystem (notcoin, hamster kombat, catizen) are all the same click-to-earn slop. the chain isn't evm compatible, so they don't even get copy-paste evm slop. there isn't a single ton-native app worth getting excited about

they even fucked up the one thing that's working, tg gifts / stickers, to the extent that 80% of the volume is now happening on p2p offchain marketplaces

it's clear telegram's distribution is extremely valuable. tg trading bots printed nine figures last year, completely ignoring ton and offering assets on other chains

the walled garden around mini apps could still work - it's not like tg users are going anywhere

but distribution is leverage, not destiny

zero times a million is still zero


```
#End of examples. Do not use information from examples in your tweets. It's just for you to understand the best style of the tweets.

"""

    USER_PROMPT = f"""
#Project feed:
{project_feed_str}
#End of project feed.
"""

    USER_PROMPT2 = f"""
For each of provided templates you need to generate a tweet.    
#Templates:
{examples_str}
IMPORTANT:  and summarize one or more topic from the project feed!
1. Full text MUST be adhere to the selected settings!
2. Each new tweet about something else.  
"""

    # 4. Инициализация LLM
    model_provider = ""
    if generation_settings.model.id in [enums.Model.GPT_4O, enums.Model.GPT_4_1, enums.Model.GPT_O4_MINI]:
        model_provider = "openai"
        api_key = settings.OPENAI_API_KEY
    elif generation_settings.model.id in [enums.Model.GEMINI_2_5_FLASH, enums.Model.GEMINI_2_5_PRO]:
        model_provider = "google_genai"
        api_key = settings.GEMINI_API_KEY
    else:
        raise NotImplementedError(f"Provider for model {generation_settings.model.id} not supported in generate_pots_examples")

    llm = init_chat_model(
        model=generation_settings.model.id.value if hasattr(generation_settings.model.id, "value") else generation_settings.model.id,
        model_provider=model_provider,
        api_key=api_key,
        temperature=0.7,
        max_tokens=2048,
    ).with_structured_output(GeneratePostsResponse)

    try:
        result: GeneratePostsResponse = await llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "user", "content": USER_PROMPT2},
        ])
    except ValidationError as e:
        # Попытка ретрая с более жёстким системным промптом
        strict_prompt = SYSTEM_PROMPT + "\nIMPORTANT: Your response MUST be valid JSON that fully complies with the schema, otherwise an error will occur."
        result: GeneratePostsResponse = await llm.ainvoke([
            {"role": "system", "content": strict_prompt},
        ])
    # 5. Пост-обработка и финальная валидация длины твитов
    cleaned: List[str] = []
    for tw in result.posts:
        tw_clean = tw.full_text
        tw_settings: PostSettings = examples[tw.settings_number-1]
        logger.info(f"Generated tweet:\nTemplate:\n {tw_settings.model_dump_json()}\nFull text:\n {tw_clean}\n\n")
        cleaned.append(tw_clean)
    # гарантия нужного количества
    if len(cleaned) != count:
        cleaned = cleaned[:count]
        while len(cleaned) < count:
            cleaned.append("")  # заполнитель, можно поднять ошибку
    return cleaned


async def generate_personalized_tweets(
    original_text: str, 
    personalization_settings: schemas.PersonalizationSettings,
    count: int = 3
) -> List[str]:
    """
    Персонализация твита на основе настроек пользователя.
    
    Принимает исходный текст твита и настройки персонализации,
    возвращает список из `count` персонализированных вариантов.
    
    Args:
        original_text: Исходный текст твита для персонализации
        personalization_settings: Настройки стиля и контента пользователя
        count: Количество вариантов для генерации (по умолчанию 3)
        
    Returns:
        Список персонализированных твитов
    """
    # --- local imports ---
    from langchain.chat_models import init_chat_model
    from pydantic import BaseModel, Field, ValidationError

    # 1. Prepare prompt based on settings
    style_instructions = []
    content_instructions = []
    
    # Style settings
    if personalization_settings.style.tonality:
        tonality_map = {
            enums.Tonality.NEUTRAL: "neutral tone",
            enums.Tonality.ENTHUSIASTIC: "enthusiastic and inspiring tone", 
            enums.Tonality.SKEPTICAL: "healthy skepticism",
            enums.Tonality.IRONIC: "light irony"
        }
        style_instructions.append(f"Use {tonality_map.get(personalization_settings.style.tonality, 'neutral tone')}")
    
    if personalization_settings.style.formality:
        formality_map = {
            enums.Formality.CASUAL: "informal, conversational style",
            enums.Formality.FORMAL: "formal, business style"
        }
        style_instructions.append(f"Write in {formality_map.get(personalization_settings.style.formality, 'informal style')}")
    
    if personalization_settings.style.perspective:
        perspective_map = {
            enums.Perspective.FIRST_PERSON: "first person (I, we)",
            enums.Perspective.THIRD_PERSON: "third person"
        }
        style_instructions.append(f"Use {perspective_map.get(personalization_settings.style.perspective, 'any perspective')}")
    
    # Content settings
    if personalization_settings.content.length:
        length_map = {
            enums.LengthOption.SHORT: "short (up to 140 characters)",
            enums.LengthOption.NORMAL: "medium (up to 220 characters)", 
            enums.LengthOption.LONG: "long (up to 280 characters)"
        }
        content_instructions.append(f"Make the tweet {length_map.get(personalization_settings.content.length, 'medium length')}")
    
    if personalization_settings.content.emoji:
        if personalization_settings.content.emoji == enums.EmojiUsage.YES:
            content_instructions.append("Add appropriate emojis")
        else:
            content_instructions.append("Do not use emojis")
    
    if personalization_settings.content.hashtags:
        if personalization_settings.content.hashtags == enums.HashtagPolicy.REQUIRED:
            content_instructions.append("Must include relevant hashtags")
        elif personalization_settings.content.hashtags == enums.HashtagPolicy.FORBIDDEN:
            content_instructions.append("Do not use hashtags")
    
    # User context section
    user_context_section = ""
    if personalization_settings.user_context:
        user_context_section = f"""
# User's personal context
{personalization_settings.user_context}
# End of user's context
Consider this context when personalizing the tweet, but DO NOT directly include or reference this information in the generated tweets.
"""
    
    # 2. Form the prompt
    SYSTEM_PROMPT = f"""
You are an expert in social media content personalization. 
Rewrite the original tweet to create {count} DIFFERENT variants, 
each preserving the main idea but adapted to the specified style and content preferences.

{user_context_section}

Style requirements: {'; '.join(style_instructions) if style_instructions else 'no specific requirements'}
Content requirements: {'; '.join(content_instructions) if content_instructions else 'no specific requirements'}

Important rules for you:
1. Rewrite the tweet — do not reply to it.
2. Follow the supplied style guidelines exactly for every version.
3. Generate several variants, making each one clearly different from both the original tweet and the other variants.
4. Each variant must be written in the SAME LANGUAGE as the ORIGINAL TWEET.
5. Return the result strictly in JSON format according to the tool schema.
    """
    USER_PROMPT = f"Original tweet to personalize:\n\"{original_text}\""
    logger.info(f"Personalization prompt: \n{SYSTEM_PROMPT}\n{USER_PROMPT}")
    
    # 3. Response schema
    class PersonalizedTweetsResponse(BaseModel):
        variants: List[str] = Field(description=f"List of {count} personalized tweet variants")
    
    # 4. Initialize LLM (using GPT-4o by default for better personalization quality)
    llm = init_chat_model(
        model=enums.Model.GPT_4O.value,
        model_provider="openai",
        api_key=settings.OPENAI_API_KEY,
        temperature=0.8,  # Higher creativity for variant diversity
        max_tokens=1024,  # Increased for better generation quality
    ).with_structured_output(PersonalizedTweetsResponse)
    
    try:
        result: PersonalizedTweetsResponse = await llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ])
    except ValidationError as e:
        # Retry with stricter system prompt
        strict_prompt = SYSTEM_PROMPT + "\nIMPORTANT: Your response MUST be valid JSON that fully complies with the schema, otherwise an error will occur."
        result: PersonalizedTweetsResponse = await llm.ainvoke([
            {"role": "system", "content": strict_prompt},
            {"role": "user", "content": USER_PROMPT},
        ])
    
    # 5. Post-processing and final tweet length validation
    cleaned: List[str] = []
    for tweet in result.variants:
        tweet_clean = tweet.strip()
        cleaned.append(tweet_clean)
    
    return cleaned


async def update_project_scores(project: models.Project, db: Session):
    """
    Обновление рейтинга для источников, которые упоминают проект.
    
    Получаем все посты проекта, которые есть.
    Обновляем очки для каждого поста.
    
    """
    
    raise NotImplementedError("Not implemented")

def extract_profile_info_from_post(post) -> tuple[Optional[str], Optional[int]]:
    try:
        user_legacy = (
            post.raw_data["core"]["user_results"]["result"]["legacy"]
        )
        profile_url = user_legacy.get("profile_image_url_https")
        followers_count = user_legacy.get("followers_count")
        return profile_url, followers_count
    except Exception:
        return None, None


def is_leaderboard_blacklisted(account: models.SocialAccount, db: Session) -> bool:
    """
    Проверяет, является ли аккаунт черным списком для лидерборда.
    """
    # если флаг is_disabled_for_leaderboard = True, то аккаунт в черном списке
    if account.is_disabled_for_leaderboard:
        return True
    
    # если логин в черном списке, то аккаунт в черном списке
    if account.social_login in leaderboard.leaderboard_logins_blacklist:
        return True
    
    # если аккаунт записан как медиа какого-то из проектов
    for project in account.projects:
        if project.status == enums.ProjectAccountStatus.MEDIA:
            return True
    
    return False


async def update_project_leaderboard(project: models.Project, db: Session):
    """
    Обновление лидерборда проекта с подробной статистикой и логированием.
    """
    from pydantic import BaseModel
    from sqlalchemy import desc
    import asyncio

    class ProjectLeaderboardUser:
        xscore: float
        new_posts_count: int
        payout: models.ScorePayout

    HOUR_PERIOD = 60*60
    NOW_TS = now_timestamp()


    # 1. Получаем дату последнего leaderboard_history (или константное время назад)
    last_history = (
        db.query(models.ProjectLeaderboardHistory)
        .filter(models.ProjectLeaderboardHistory.project_id == project.id)
        .order_by(desc(models.ProjectLeaderboardHistory.created_at))
        .first()
    )
    if last_history:
        period_start = last_history.end_ts+1
        logger.info(f"[Leaderboard] Last history found: {period_start}")
    else:
        period_start = now_timestamp() - HOUR_PERIOD*4 # 4 hours
        logger.info(f"[Leaderboard] No history found, using last 4h: {period_start}")
        
    period_end = NOW_TS
    # Ограничиваем период, чтобы не обрабатывать сразу слишком много постов
    if period_end - period_start > HOUR_PERIOD*24:
        period_end = min(period_start + HOUR_PERIOD*24, NOW_TS)
        logger.info(f"[Leaderboard] Period is too long, limiting to 24 hours: {period_end}")
        
        
    logger.info(f"[Leaderboard] Period: {period_start} - {period_end}")
    period_seconds = period_end - period_start

    # 2. Получаем посты с упоминанием проекта, у которых есть статистика за период
    posts_generator = crud.get_updated_posts(db, project, period_start, period_end, batch_size=500)
    logger.info(f"[Leaderboard] Posts fetched")

    # 3. Для каждого поста ищем старую и новую статистику, считаем engagement
    user_stats: dict[int, ProjectLeaderboardUser] = {}
    total_stats_processed = 0
    posts_with_engagement = 0
    for post in posts_generator:
        # пропускаем посты, автор которых в черном списке по лидерборду
        if is_leaderboard_blacklisted(post.account, db):
            continue
        
        # Найти свежую статистику за период (created_at >= period_start)
        stats_in_period = [s for s in post.statistic if s.created_at >= period_start]
        total_stats_processed += len(stats_in_period)
        if not stats_in_period:
            continue  # пост не обновлялся за период
        new_stats = max(stats_in_period, key=lambda s: s.created_at)
        # Найти самую свежую статистику до периода (created_at < period_start)
        stats_before = [s for s in post.statistic if s.created_at < period_start]
        if stats_before:
            old_stats = max(stats_before, key=lambda s: s.created_at)
            old_views = old_stats.views
            old_likes = old_stats.likes
            old_reposts = old_stats.reposts
            old_comments = old_stats.comments
        else:
            old_views = old_likes = old_reposts = old_comments = 0
        # Считаем engagement
        delta_views = max(0, new_stats.views - old_views)
        delta_likes = max(0, new_stats.likes - old_likes)
        delta_reposts = max(0, new_stats.reposts - old_reposts)
        delta_comments = max(0, new_stats.comments - old_comments)
        engagement = calculate_post_engagement_score(delta_views, delta_likes, delta_reposts, delta_comments)
        if engagement == 0:
            continue
        posts_with_engagement += 1
        acc_id = post.account_id

        
        if acc_id not in user_stats:
            # Получаем xscore (или 50)
            # TODO: скоры желательно получать по API
            xscore = post.account.twitter_scout_score if post.account.twitter_scout_score is not None else 1
            # дополнительно ограничиваем снизу, чтобы не было 0
            xscore = max(xscore, 1)
            user_stats[acc_id] = ProjectLeaderboardUser()
            user_stats[acc_id].xscore = xscore
            user_stats[acc_id].payout = models.ScorePayout(
                project_id=project.id,
                social_account_id=acc_id,
                engagement=engagement,
                new_posts_count=0,
            )
        user_stats[acc_id].payout.engagement += engagement
        # Если данный пост был добавлен в базу за период, то нужно его учесть.
        # !Важно, нужно именно created_at, а не posted_at, тк posted_at может быть в прошлом (из-за того, что пост не сразу попадает в базу)
        if post.created_at >= period_start:
            user_stats[acc_id].payout.new_posts_count += 1

    if not user_stats:
        logger.warning(f"[Leaderboard] No user engagement for leaderboard update for project {project.id}")
        logger.info(f"[Leaderboard] Stats processed: {total_stats_processed}, posts with engagement: {posts_with_engagement}")

    # 4. Считаем суммарные xscore и engagement
    sum_xscore = 0
    sum_engagement = 0
    for user in user_stats.values():
        sum_xscore += user.xscore
        sum_engagement += user.payout.engagement
    logger.info(f"[Leaderboard] Users: {len(user_stats)}, sum_xscore: {sum_xscore}, sum_engagement: {sum_engagement}")
    if sum_engagement > 0:

        # 5. Считаем period_reward (при условии, что за 24 часа должны раздать 1000 base_score)
        period_reward = 1000 * (period_seconds / 86400)
        logger.info(f"[Leaderboard] period_reward: {period_reward:.2f} for {period_seconds} seconds")

        # 6. Для каждого автора считаем XUSD, EDS, mindshare, base_score, score
        total_mindshare = 0
        total_base_score = 0
        total_score = 0
        for social_account_id, user in user_stats.items():
            XUSD = user.xscore / sum_xscore
            EDS = user.payout.engagement / sum_engagement
            user.payout.mindshare = 0.5 * XUSD + 0.5 * EDS
            base_score = user.payout.mindshare * period_reward
            user.payout.base_score = base_score
            score = base_score
            # Если есть юзер у этого соц аккаунта, то нужно учитывать мультипликаторы и считать доп информацию
            app_user = await crud.get_user_by_social_account_id(db, social_account_id)
            if app_user:
                # подсчитываем мультипликаторы
                # получаем последний Payout для этого юзера
                last_payout = await crud.get_last_payout_by_social_account_id(db, social_account_id, project.id)
                if not last_payout:
                    # если еще небыло выплат по этому проекту для юзера, значит это первый пост
                    # TODO: не уверен, что нужно устанавливать именно это время
                    user.payout.first_post_at = NOW_TS
                    user.payout.last_post_at = NOW_TS
                    user.payout.weekly_streak_start_at = NOW_TS
                    user.payout.loyalty_points = 0
                    user.payout.min_loyalty = 0
                else:
                    # сохраняем прошлые значения
                    user.payout.first_post_at = last_payout.first_post_at if last_payout.first_post_at is not None else NOW_TS
                    user.payout.last_post_at = last_payout.last_post_at if last_payout.last_post_at is not None else NOW_TS
                    user.payout.weekly_streak_start_at = last_payout.weekly_streak_start_at if last_payout.weekly_streak_start_at is not None else NOW_TS
                    user.payout.loyalty_points = last_payout.loyalty_points if last_payout.loyalty_points is not None else 0
                    user.payout.min_loyalty = last_payout.min_loyalty if last_payout.min_loyalty is not None else 0
            
                ### рассчитываем бонусы
                ## бонус за первую неделю
                # рассчитываем бонус за первую неделю
                week_period = 60*60*24*7
                if user.payout.first_post_at >= NOW_TS - week_period:
                    score += base_score*9

                ## бонус за стрик
                # сбрасываем стрик, если не было упоминания об проекте за неделю
                if user.payout.last_post_at < NOW_TS - week_period:
                    user.payout.weekly_streak_start_at = NOW_TS
                
                # считаем длину стрика
                weeks_since_streak_start = (NOW_TS - user.payout.weekly_streak_start_at) // week_period
                multiplier = utils_base.streak_to_multiplier(weeks_since_streak_start)
                score += (base_score*multiplier) - base_score

                ## бонус за лояльность
                # снимаем очки лояльности за каждую целую неделю некативности.
                week_penalty = 0.1
                penalty_multiplier = 1-week_penalty
                weeks_since_last_post = (NOW_TS - user.payout.last_post_at) // week_period
                user.payout.loyalty_points = max(
                    user.payout.min_loyalty, 
                    user.payout.loyalty_points*(penalty_multiplier**weeks_since_last_post)
                )
                # а затем добавляем loyalty_points на основе mindshare
                mindshare_to_loyalty_points = {
                    0: 0.75,
                    0.5: 1,
                    2: 1.5,
                    5: 2,
                    10: 4,
                    12.5: 4.5,
                    15: 5,
                    17.5: 5.5,
                    20: 6
                }
                max_loyalty_points = 100
                max_available_points = 0
                for mindshare, points in mindshare_to_loyalty_points.items():
                    if user.payout.mindshare >= mindshare:
                        max_available_points = max(max_available_points, points)
                # плюсуем так, чтобы не превысить max_loyalty_points
                points_to_add = max_available_points*(period_seconds / 86400)
                user.payout.loyalty_points = min(
                    user.payout.loyalty_points+points_to_add,
                    max_loyalty_points
                )
                # начисляем бонус за лояльность
                
                # если пользователь имеет подписку Ultra, то его лояльность увеличивается на 30 очков
                # но только в момент начисления бонуса за лояльность
                user_actual_loyalty = user.payout.loyalty_points
                user_last_plan_check = crud.get_user_last_plan_check(db, app_user.id)
                if user_last_plan_check:
                    if user_last_plan_check.user_plan == enums.SubscriptionPlanType.ULTRA:
                        user_actual_loyalty += 30
                
                loyalty_multiplier = utils_base.loyalty_to_multiplier(user_actual_loyalty)
                score += base_score*loyalty_multiplier - base_score
                
                # при необходимости обновляем min_loyalty
                if user.payout.loyalty_points == 100:
                    user.payout.min_loyalty = 50
                elif user.payout.loyalty_points >= 80:
                    user.payout.min_loyalty = max(user.payout.min_loyalty, 40)
                elif user.payout.loyalty_points >= 50:
                    user.payout.min_loyalty = max(user.payout.min_loyalty, 30)

                # обновляем дату последнего поста, если были новые посты за этот период
                if user.payout.new_posts_count > 0:
                    user.payout.last_post_at = NOW_TS
                    
                logger.info(f"[Leaderboard] User {app_user.id} score: {score:.2f}, mindshare: {user.payout.mindshare:.4f}, base_score: {user.payout.base_score:.2f}, engagement: {user.payout.engagement:.2f}, new_posts_count: {user.payout.new_posts_count}, first_post_at: {user.payout.first_post_at}, last_post_at: {user.payout.last_post_at}, weekly_streak_start_at: {user.payout.weekly_streak_start_at}, loyalty_points: {user.payout.loyalty_points}, min_loyalty: {user.payout.min_loyalty}")

            user.payout.score = score
            
            total_mindshare += user.payout.mindshare
            total_base_score += user.payout.base_score
            total_score += user.payout.score

        logger.info(f"[Leaderboard] Total mindshare: {total_mindshare:.4f}, total base_score: {total_base_score:.2f}, total score: {total_score:.2f}")
    else:
        logger.warning(f"[Leaderboard] No user engagement for leaderboard update for project {project.id}")

    # 7. Сохраняем ScorePayout и ProjectLeaderboardHistory
    try:
        history = models.ProjectLeaderboardHistory(
            project_id=project.id,
            start_ts=period_start,
            end_ts=period_end,
            created_at=period_end,
        )
        db.add(history)
        db.flush()  # чтобы получить history.id

        if sum_engagement > 0 and len(user_stats) > 0:
            for user in user_stats.values():
                user.payout.project_leaderboard_history_id = history.id
                user.payout.created_at = period_end
                db.add(user.payout)
        db.commit()
        logger.info(f"[Leaderboard] Leaderboard updated for project {project.id}: ...")
    except Exception as e:
        db.rollback()
        logger.error(f"[Leaderboard] Error saving payouts/history for project {project.id}: {e}\n{traceback.format_exc()}")
        return


async def cleanup_old_posts(db: Session)->int:
    """
    Очищает старые посты после обновлений, чтобы не засорять базу.
    """
    cutoff_timestamp = now_timestamp() - settings.POST_TO_TRASH_LIFETIME_SECONDS
    return await crud.delete_old_posts(db, cutoff_timestamp)


async def master_update(db: Session):
    """
    Единый метод для обновления всех данных приложения.
    
    1. Безопасно обновляет посты по всем проектам.
    2. Обновляет лидерборды по нужным проектам.
    3. Очищает старые посты после обновлений, чтобы не засорять базу.
    
    Вызывается из celery task и cli скрипта.
    """
    from_ts = now_timestamp() - settings.POST_SYNC_PERIOD_SECONDS
    
    # проекты для обновления
    projects = db.query(models.Project).all()
    for project in projects:
        try:
            # обновляем посты фида (от всех источников)
            feed_updated = await update_project_feed(project, from_ts, db)
            # обновляем news посты (от офф источников)
            news_updated = await update_project_news(project, from_ts, db)
            
            if not (feed_updated or news_updated):
                logger.info(f"[MasterUpdate] No updates for project {project.id}")
                continue
            # обновляем кеш фида (всегда, если были обновления)
            try:
                await get_feed(project, db, force_rebuild=True)
                logger.info(f"[MasterUpdate] Feed cache updated for project {project.id}")
            except Exception as e:
                logger.error(f"[MasterUpdate] Error updating feed cache for project {project.id}: {e}")
                
            if project.is_leaderboard_project:
                # обновляем лидерборд
                try:
                    await update_project_leaderboard(project, db)
                    logger.info(f"[MasterUpdate] Leaderboard updated for project {project.id}")
                except Exception as e:
                    logger.error(f"[MasterUpdate] Error updating leaderboard for project {project.id}: {e}")
                
                # обновляем кеш лидерборда
                # для любого из периодов (обновляет все кеши)
                try:
                    await get_leaderboard(project, cache_service.LeaderboardPeriod.ALL_TIME, db, force_rebuild=True)
                except Exception as e:
                    logger.error(f"[MasterUpdate] Error updating leaderboard cache for project {project.id}: {e}")
              
        except Exception as e:
            logger.error(f"[MasterUpdate] Error updating project {project.id}. Details: {e} {traceback.format_exc()}")
            
    # очищаем старые посты
    deleted_count = await cleanup_old_posts(db)
    logger.info(f"[MasterUpdate] Deleted {deleted_count} old posts")
    

def update_users_xscore(db: Session):
    """
    Обновляет xscore для всех пользователей, чьи аккаунты упоминают лидербордские проекты и их xcore не установлен.
    """
    
    # получаем все аккаунты, у которых есть хотя бы один пост с упоминанием любого проекта с включенным лидербордом.
    social_accounts = (
        db.query(models.SocialAccount)
        .join(models.SocialPost, models.SocialAccount.id == models.SocialPost.account_id)
        .join(models.ProjectMention, models.SocialPost.id == models.ProjectMention.post_id)
        .join(models.Project, models.ProjectMention.project_id == models.Project.id)
        .filter(models.Project.is_leaderboard_project == True)
        .distinct()
        .all()
    )
    
    # обновляем xscore для этих пользователей
    for social_account in social_accounts:
        if social_account.twitter_scout_score is not None:
            continue
        try:
            user_xscore = get_social_account_xscore(social_account)
            social_account.twitter_scout_score = user_xscore
            social_account.twitter_scout_score_updated_at = now_timestamp()
            db.add(social_account)
            db.commit()
            db.refresh(social_account)
            logger.info(f"[XScore update] Updated xscore for social account {social_account.id}: {user_xscore}")
        except Exception as e:
            logger.error(f"[XScore update] Error updating xscore for social account {social_account.id}: {e}")
        pass
    
    logger.info(f"[XScore update] Updated {len(social_accounts)} social accounts")


def update_users_xscore_with_linked_accounts(db: Session):
    """
    Обновляет xscore для всех социальных аккаунтов, чьи логины совпадают с установленными логинами пользователей.
    Score устанавливается только если он не был установлен ранее.
    """
    # 1. Извлечь все уникальные twitter_login из пользователей
    logins_tuples = db.query(models.User.twitter_login).filter(models.User.twitter_login.isnot(None)).distinct().all()
    logins = [login for (login,) in logins_tuples]

    if not logins:
        logger.info("[XScore update linked] No users with linked twitter accounts found.")
        return

    logger.info(f"[XScore update linked] Found {len(logins)} unique linked twitter accounts to check.")
    updated_count = 0

    # 2. Для каждого логина найти соответствующий social_account и обновить его
    for login in logins:
        social_account = db.query(models.SocialAccount).filter(models.SocialAccount.social_login == login).first()

        if social_account:
            # Пропускаем, если скор обновлялся меньше месяца назад
            one_month_ago_ts = now_timestamp() - (30 * 24 * 60 * 60)
            if social_account.twitter_scout_score_updated_at and social_account.twitter_scout_score_updated_at > one_month_ago_ts:
                continue

            try:
                user_xscore = get_social_account_xscore(social_account)
                if user_xscore is not None:
                    social_account.twitter_scout_score = user_xscore
                    social_account.twitter_scout_score_updated_at = now_timestamp()
                    db.add(social_account)
                    updated_count += 1
                    logger.info(f"[XScore update linked] Set xscore for social account {social_account.id} ({social_account.social_login}): {user_xscore}")
                else:
                    logger.warning(f"[XScore update linked] Could not get xscore for {login}")
            except Exception as e:
                logger.error(f"[XScore update linked] Error updating xscore for social account with login {login}: {e}")
        else:
            logger.warning(f"[XScore update linked] Social account not found for login {login}")

    if updated_count > 0:
        try:
            db.commit()
            logger.info(f"[XScore update linked] Committed updates for {updated_count} social accounts.")
        except Exception as e:
            db.rollback()
            logger.error(f"[XScore update linked] Error committing updates: {e}")

    logger.info(f"[XScore update linked] Finished. Updated {updated_count} social accounts.")


def get_social_account_xscore(social_account: models.SocialAccount) -> float:
    """
    Получает xscore для пользователя.
    
    https://api.tweetscout.io/v2/score/{user_handle}
    """
    if not settings.TWITTER_SCOUT_API_KEY:
        raise Exception("TWITTER_SCOUT_API_KEY is not set")
    
    url = f"https://api.tweetscout.io/v2/score/{social_account.social_login}"
    response = requests.get(url, headers={"ApiKey": f"{settings.TWITTER_SCOUT_API_KEY}", "Accept": "application/json"})
    if response.status_code == 200:
        return response.json()["score"]
    else:
        return None
    

async def get_leaderboard(project: models.Project, period: cache_service.LeaderboardPeriod, db: Session, force_rebuild: bool = False) -> List[dict]:
    """
    Получить лидерборд проекта за указанный период с кешированием.
    Если force_rebuild=True — всегда пересчитывает и обновляет кеш.
    """
    
    leaderboard_cache = LeaderboardCacheService()
    if not force_rebuild:
        cached = await leaderboard_cache.get_leaderboard(project.id, period)
        if cached is not None:
            logger.info(f"[Leaderboard] Cache hit for project {project.name}, period {period}")
            return cached
    logger.info(f"[Leaderboard] Cache miss for project {project.name}, period {period}")
    
    
    # --- Пересчёт ---
    # Определяем from_ts для периода
    now_ts = now_timestamp()
    # Получаем всю историю
    histories = (
        db.query(models.ProjectLeaderboardHistory)
        .options(joinedload(models.ProjectLeaderboardHistory.scores).joinedload(models.ScorePayout.social_account))
        .filter(models.ProjectLeaderboardHistory.project_id == project.id)
        .order_by(models.ProjectLeaderboardHistory.created_at)
        .all()
    )
    
    # для начала, считаем общие данные за все время для каждого пользователя
    class AllTimeUserHistory:
        score: float = 0
        posts: int = 0
        avatar_url: Optional[str] = None
        followers: Optional[int] = None
        is_connected: bool = False
        name: Optional[str] = None
        login: Optional[str] = None
    
    
    all_time_users: dict[int, AllTimeUserHistory] = {}
    # считаем общие данные за все время для каждого пользователя
    for history in histories:
        for payout in history.scores:
            if payout.social_account_id not in all_time_users:
                all_time_users[payout.social_account_id] = AllTimeUserHistory()
                
                social_account = payout.social_account
                if social_account is not None:
                    all_time_users[payout.social_account_id].name = social_account.name or social_account.social_login
                    all_time_users[payout.social_account_id].login = social_account.social_login
                    all_time_users[payout.social_account_id].avatar_url = social_account.last_avatar_url
                    all_time_users[payout.social_account_id].followers = social_account.last_followers_count
                    # TODO: это не лучший способ, но пока так. В идеале, нужно удостовериться, что есть пользователь с таким подключенным аккаунтом
                    all_time_users[payout.social_account_id].is_connected = social_account.twitter_scout_score_updated_at is not None
            all_time_users[payout.social_account_id].score += payout.score if payout.score is not None else 0
            all_time_users[payout.social_account_id].posts += payout.new_posts_count if payout.new_posts_count is not None else 0
    
    # TODO: для каждого пользователя получаем присутствие в проекте
    # окончание периода - это максимальный end_ts среди всех периодов
    end_ts = max(history.end_ts for history in histories)
    first_ts = min(history.start_ts for history in histories)
    
    ONE_DAY = 86400
    
    # для каждого периода считаем данные
    for cached_preriod in cache_service.LeaderboardPeriod:
        if cached_preriod == cache_service.LeaderboardPeriod.ALL_TIME:
            start_ts = first_ts
        elif cached_preriod == cache_service.LeaderboardPeriod.ONE_DAY:
            start_ts = end_ts - ONE_DAY
        elif cached_preriod == cache_service.LeaderboardPeriod.ONE_WEEK:
            start_ts = end_ts - ONE_DAY*7
        elif cached_preriod == cache_service.LeaderboardPeriod.ONE_MONTH:
            start_ts = end_ts - ONE_DAY*30
        else:
            raise ValueError(f"Unknown leaderboard period: {cached_preriod}")
        start_ts = max(start_ts, first_ts)
        # для подсчета взвешенного mindshare
        all_period_seconds = end_ts - start_ts
        def timespan_to_str(ts: int) -> str:
            return f"{datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}"
        logger.info(f"[Leaderboard] Подсчет для периода {cached_preriod}: {timespan_to_str(start_ts)} - {timespan_to_str(end_ts)}, {all_period_seconds/(60*60)} часов")
        
        # место для хранения данных за период
        leaderboard_users: dict[int, schemas.LeaderboardUser] = {}
        # проходим по истории
        for history in histories:
            # пропускаем истории, которые не входят в период
            if history.created_at < start_ts:
                continue
            # время, которое охватывает эта история
            # TODO: первый период обычно нужно считать не полным и обрезать его period и полученные за этот период скоры, майндшер и тд
            history_period_seconds = history.end_ts - max(history.start_ts, start_ts)
            # TODO: считать mindshare_delta
            for payout in history.scores:
                if payout.social_account_id not in leaderboard_users:
                    user_all_time_data = all_time_users.get(payout.social_account_id, None)
                    if user_all_time_data is None:
                        # такого не должно быть, но если что, то пропускаем
                        logger.error(f"[Leaderboard] No all time data for social account {payout.social_account_id}")
                        continue
                    leaderboard_users[payout.social_account_id] = schemas.LeaderboardUser(
                        avatar_url=user_all_time_data.avatar_url,
                        name=user_all_time_data.name or "",
                        login=user_all_time_data.login or "",
                        followers=user_all_time_data.followers,
                        mindshare=0,
                        mindshare_delta=0,
                        engagement=0,
                        scores=0,
                        scores_all_time=user_all_time_data.score,
                        posts_period=0,
                        posts_all_time=user_all_time_data.posts,
                        is_connected=user_all_time_data.is_connected,
                    )
                leaderboard_users[payout.social_account_id].scores += payout.score if payout.score is not None else 0
                leaderboard_users[payout.social_account_id].posts_period += payout.new_posts_count if payout.new_posts_count is not None else 0
                leaderboard_users[payout.social_account_id].engagement += payout.engagement if payout.engagement is not None else 0
                leaderboard_users[payout.social_account_id].mindshare += payout.mindshare*(history_period_seconds/all_period_seconds) if payout.mindshare is not None else 0

        #TODO: чисто для отладки отображаем статистику по подсчету
        leaderboard_users = list(leaderboard_users.values())
        leaderboard_users.sort(key=lambda x: x.mindshare, reverse=True)
        #for user in leaderboard_users:
        #    logger.info(f"[Leaderboard] User {user.login}: mindshare={user.mindshare}, scores={user.scores}, posts_period={user.posts_period}, engagement={user.engagement}")
        # Сериализация (pydantic -> dict)
        users_data = [u.model_dump() for u in leaderboard_users]
        await leaderboard_cache.set_leaderboard(project.id, cached_preriod, users_data)
        logger.info(f"[Leaderboard] Cache set for project {project.name}, period {cached_preriod} with {len(users_data)} users")
    
    # после пересчета снова пытаемся получить кеш
    cached = await leaderboard_cache.get_leaderboard(project.id, period)
    if cached is not None:
        return cached
    logger.error(f"[Leaderboard] Error updating leaderboard cache for project {project.id}, period {period}")
    # если кеш не получился, то возвращаем пустой список
    return []
    

async def get_feed(project: models.Project, db: Session, force_rebuild: bool = False) -> List[schemas.Post]:
    """
    Получить топ-100 постов по engagement за сутки для проекта с кешированием.
    Если force_rebuild=True — всегда пересчитывает и обновляет кеш.
    """
    feed_cache = FeedCacheService()
    # если кеш не нужно пересчитывать, то возвращаем кешированные данные
    if not force_rebuild:
        cached = await feed_cache.get_feed(project.id)
        if cached is not None:
            # Десериализация
            logger.info(f"[Feed] Cache hit for project {project.name}")
            return [schemas.Post(**p) for p in cached]
    logger.info(f"[Feed] Cache miss for project {project.name}")
    # --- Пересчёт ---
    posts = get_top_engagement_posts(project, db, limit=100, period=86400)
    posts_schemas = convert_posts_to_schemas(posts)
    # Сериализация
    posts_data = [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in posts_schemas]
    await feed_cache.set_feed(project.id, posts_data)
    return posts_schemas
    
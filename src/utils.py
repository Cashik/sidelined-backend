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
from sqlalchemy.orm import Session, aliased
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
import requests

from src import schemas, enums, models, exceptions, crud
from src.config.settings import settings
from src.config.subscription_plans import subscription_plans
from src.config.ai_models import all_ai_models as ai_models
from src.services.web3_service import Web3Service
from src.services.prompt_service import PromptService
from src.services import x_api_service
from src.config.mcp_servers import mcp_servers as mcp_servers_list
from src.utils_base import now_timestamp, parse_date_to_timestamp
import inspect
from collections import defaultdict

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


async def update_project_feed(project: models.Project, from_timestamp: int, db: Session) -> bool:
    """
    Обновление фида проекта.
    Тут нам важно, чтобы в твите было упоминание проекта и твит имел достаточное количество лайков.
    """
    keywords = project.keywords.split(";")
    keywords = [f'"{kw}"' for kw in keywords if kw]
    keywords_query = " OR ".join(keywords)
    query = f"({keywords_query}) min_faves:{settings.DEFAULT_MINIMAL_LIKES_TO_SEARCH_TWEETS}"
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

        source_id = post.core.user_results.result.rest_id
        source_db: models.SocialAccount | None = await crud.get_social_account_by_id(source_id, db)
        if not source_db:
            # создаем источник
            source_data: XApiService.User = post.core.user_results.result
            if not hasattr(source_data, 'legacy'):
                logger.warning(f"Skipping tweet due to missing source legacy data: {source_data}")
                return

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
        model=default_model
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
    SYSTEM_PROMPT = """
You are a social media content creator AI agent.
Your task it to analyze some project feed (in X social network) and generate tweets about the project like a human based on one of the given templates.

#Important rules:
- Return the result strictly in JSON format according to the tool schema, without any additional text.
- Analyze the project's feed to highlight all topics or lastests news for a new tweets.
- Write in first or third person, as if you are a regular user.
- You ARE NOT THE PART OF THE PROJECT.
- Do not mention the project directly (the system will do it) and do not mention anyone.
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

the moment they go mobile with that same experience, it’s the end of cex

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

*Posts pic of BTC tape* — “Hmmm interesting. Looks healthy! Think we have a real shot to go up from here! Things are finally looking promising!”

What?

Didn’t you JUST dedicate weeks of your life to bearposting?

What do you mean it “looks good now” when we’re about to break ATH you stupid fuck?


—-----

100 supply Overworld Keys went to 18e in the worst bear market for NFTs for a game that literally never existed and a company that didn’t deliver *anything*

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

Big news from @peaq that’s been quietly in the works for a while:

They’ve officially launched the world’s first Machine Economy Free Zone (MEFZ) in the UAE - a real-world sandbox to test, deploy, and scale machine economy applications with full regulatory and investment backing.

For those who don’t realize how massive this is:

This marks another huge leap forward in peaq’s mission to build the universal coordination layer for DePINs, machines, and robots - now grounded in a physical zone with clear regulation, real adoption paths, R&D infrastructure, and investment / liquidity pipelines.

In my opinion, this just made peaq orders of magnitude more relevant for anyone building in DePIN, DePAI, robotics, and machine economy verticals.

Expecting the peaqosystem to experience explosive growth from here - with fresh use cases, stronger network effects, and more real-world traction than ever.

And it’ll be all powered by $PEAQ at the core.


—----


In case you missed it: the big money is coming for your $ETH - aggressively.

Yesterday alone saw over $125M in net inflows into spot ETH ETFs, which marks the largest daily inflow in the past 4 months.

But more impressively, ETH ETFs are now on a record-breaking streak of consecutive daily inflows, totaling over $1 billion in 17 days. 

That’s not just bullish - that’s conviction.

For the first time ever, total assets under management have now exceeded $10 billion.

Macro. Narratives. Fundamentals. Ecosystem. Execution.

Everything is aligning for Ethereum, and smart money is clearly watching.

Wouldn’t be surprised if June/July ends up being the last time we ever see ETH under $3K.

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
    
    # 2. Form the prompt
    SYSTEM_PROMPT = f"""
You are an expert in social media content personalization. 
Rewrite the original tweet to create {count} DIFFERENT variants, 
each preserving the main idea but adapted to the specified style and content preferences.
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
        tweet_clean = tweet.strip().replace("\n", " ")
        if len(tweet_clean) > 280:
            tweet_clean = tweet_clean[:277] + "…"
        cleaned.append(tweet_clean)
    
    # Guarantee the required number of variants
    if len(cleaned) != count:
        cleaned = cleaned[:count]
        while len(cleaned) < count:
            # If not enough variants, duplicate the last one with small modifications
            if cleaned:
                base_tweet = cleaned[-1]
                # Simple modification to create a variant
                if "!" not in base_tweet and len(base_tweet) < 279:
                    cleaned.append(base_tweet + "!")
                elif "." in base_tweet:
                    cleaned.append(base_tweet.replace(".", "..."))
                else:
                    cleaned.append(base_tweet)
            else:
                cleaned.append(original_text)  # Fallback to original text
    
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

def build_leaderboard_users(histories: list[models.ProjectLeaderboardHistory], from_ts) -> list[schemas.LeaderboardUser]:
    """
    Формирует список LeaderboardUser по истории лидерборда проекта.
    - mindshare: среднее арифметическое по всем историям (если нет выплаты — 0)
    - scores: сумма всех выплат
    - avatar_url, followers: из самого свежего поста (если есть)
    - name, login: из SocialAccount
    """
    if not histories:
        return []
    
    total_period_seconds = 0
    for history in histories:
        total_period_seconds += history.end_ts - max(history.start_ts, from_ts)

    # Собираем все уникальные аккаунты
    accounts = {}
    posts_by_account = defaultdict(list)
    for history in histories:
        period_seconds = history.end_ts - max(history.start_ts, from_ts)
        for payout in history.scores:
            acc = payout.social_account
            # создаем аккаунт, если его нет
            if acc.id not in accounts:
                accounts[acc.id] = schemas.LeaderboardUser(
                    avatar_url=None,
                    name=acc.name or acc.social_login or f"id{acc.id}",
                    login=acc.social_login or f"id{acc.id}",
                    followers=None,
                    mindshare=0.0,
                    scores=0.0
                )
            # суммируем mindshare и scores
            accounts[acc.id].mindshare += (payout.mindshare*period_seconds)/total_period_seconds
            accounts[acc.id].scores += payout.score
            # собираем посты
            posts_by_account[acc.id].extend(acc.posts)
    
    logger.info(f"Sum of mindshare: {sum(acc.mindshare for acc in accounts.values())}")
    # Сортируем по scores убыванию
    result = list(accounts.values())
    # --- ДОБАВЛЯЕМ avatar_url и followers из самого свежего поста ---
    for acc_id, acc in accounts.items():
        user_posts = posts_by_account.get(acc_id, [])
        if user_posts:
            latest_post = max(user_posts, key=lambda p: p.posted_at)
            profile_url, followers_count = extract_profile_info_from_post(latest_post)
        else:
            profile_url, followers_count = None, None
        acc.avatar_url = profile_url
        acc.followers = followers_count
    result.sort(key=lambda u: u.scores, reverse=True)
    return result


async def update_project_leaderboard(project: models.Project, db: Session):
    """
    Обновление лидерборда проекта с подробной статистикой и логированием.
    """
    from pydantic import BaseModel
    from sqlalchemy import desc
    import asyncio

    class ProjectLeaderboardUser(BaseModel):
        social_account_id: int
        xscore: float = 0.0
        base_engagement: float = 0.0
        mindshare: float = 0.0
        base_score: float = 0.0
        score: float = 0.0

    # 1. Получаем дату последнего leaderboard_history (или сутки назад)
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
        period_start = now_timestamp() - 86400
        logger.info(f"[Leaderboard] No history found, using last 24h: {period_start}")
    period_end = now_timestamp()
    period_seconds = period_end - period_start

    # 2. Получаем ВСЕ посты с упоминанием проекта, у которых есть статистика за период
    posts: list[models.SocialPost] = await crud.get_project_feed(project, settings.POST_SYNC_PERIOD_SECONDS, db)
    logger.info(f"[Leaderboard] Posts fetched: {len(posts)}")
    if not posts:
        logger.info(f"[Leaderboard] No posts for leaderboard update for project {project.id}")
        return

    # 3. Для каждого поста ищем старую и новую статистику, считаем engagement
    user_stats: dict[int, ProjectLeaderboardUser] = {}
    total_stats_processed = 0
    posts_with_engagement = 0
    for post in posts:
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
            xscore = post.account.twitter_scout_score if post.account.twitter_scout_score is not None else 50.0
            user_stats[acc_id] = ProjectLeaderboardUser(social_account_id=acc_id, xscore=xscore)
        user_stats[acc_id].base_engagement += engagement

    if not user_stats:
        logger.warning(f"[Leaderboard] No user engagement for leaderboard update for project {project.id}")
        logger.info(f"[Leaderboard] Stats processed: {total_stats_processed}, posts with engagement: {posts_with_engagement}")
        return

    # 4. Считаем суммарные xscore и engagement
    sum_xscore = 0
    sum_engagement = 0
    for u in user_stats.values():
        sum_xscore += u.xscore
        sum_engagement += u.base_engagement
    logger.info(f"[Leaderboard] Users: {len(user_stats)}, sum_xscore: {sum_xscore}, sum_engagement: {sum_engagement}")
    if sum_xscore == 0 or sum_engagement == 0:
        logger.warning(f"[Leaderboard] Zero xscore or engagement for leaderboard update for project {project.id}")
        return

    # 5. Считаем period_reward (при условии, что за 24 часа должны раздать 1000 base_score)
    period_reward = 1000 * (period_seconds / 86400)
    logger.info(f"[Leaderboard] period_reward: {period_reward:.2f} for {period_seconds} seconds")

    # 6. Для каждого автора считаем XUSD, EDS, mindshare, base_score, score
    total_mindshare = 0
    total_base_score = 0
    total_score = 0
    for u in user_stats.values():
        XUSD = u.xscore / sum_xscore
        EDS = u.base_engagement / sum_engagement
        u.mindshare = 0.5 * XUSD + 0.5 * EDS
        u.base_score = u.mindshare * period_reward
        u.score = u.base_score * 1  # TODO: учесть мультипликаторы юзера
        total_mindshare += u.mindshare
        total_base_score += u.base_score
        total_score += u.score

    logger.info(f"[Leaderboard] Total mindshare: {total_mindshare:.4f}, total base_score: {total_base_score:.2f}, total score: {total_score:.2f}")

    # 7. Сохраняем ScorePayout и ProjectLeaderboardHistory
    
    history = models.ProjectLeaderboardHistory(
        project_id=project.id,
        start_ts=period_start,
        end_ts=period_end,
        created_at=period_end,
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    payouts = []
    for u in user_stats.values():
        payout = models.ScorePayout(
            project_id=project.id,
            social_account_id=u.social_account_id,
            project_leaderboard_history_id=history.id,
            score=u.score,
            engagement=u.base_engagement,
            mindshare=u.mindshare,
            base_score=u.base_score,
            created_at=period_end,
        )
        db.add(payout)
        payouts.append(payout)
    db.commit()
    
    logger.info(f"[Leaderboard] Leaderboard updated for project {project.id}: {len(payouts)} payouts, period {period_start} - {period_end}")
    logger.info(f"[Leaderboard] Posts processed: {len(posts)}, stats processed: {total_stats_processed}, posts with engagement: {posts_with_engagement}, users: {len(user_stats)}")


async def cleanup_old_posts(db: Session):
    """
    Очищает старые посты после обновлений, чтобы не засорять базу.
    """
    cutoff_timestamp = now_timestamp() - settings.POST_TO_TRASH_LIFETIME_SECONDS
    await crud.delete_old_posts(db, cutoff_timestamp)


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
            
            if project.is_leaderboard_project:
                # обновляем лидерборд
                await update_project_leaderboard(project, db)
                logger.info(f"[MasterUpdate] Leaderboard updated for project {project.id}")
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
    
    
    # получаем все посты с упоминанием проекта лидерборда
    
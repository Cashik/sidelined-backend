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
from src.config.ai_models import all_ai_models as ai_models
from src.services.web3_service import Web3Service
from src.services.prompt_service import PromptService
from src.services import x_api_service
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


async def update_project_data(project: models.Project, from_timestamp: int, db: Session) -> bool:
    """
    Обновление данных проекта.
    
    Находит новые твиты и парсит их, чтобы отразить в базе данных.
    
    from_timestamp - timestamp в секундах, с которого нужно искать твиты. Это либо время создания проекта, либо максимальный срок, который мы учитываем в статистике.
    """
    # сервис для работы с x api
    x_api_service = XApiService()
    keywords = project.keywords.split(";")
    keywords = [f'"{kw}"' for kw in keywords if kw]
    keywords_query = " OR ".join(keywords)
    query = f"({keywords_query}) min_faves:{settings.POST_SYNC_LIKES_COUNT_MINIMAL}"
    tweets_added_count = 0
    
    try:
        # Получаем твитов сколько сможем
        response: XApiService.FeedResponse = await x_api_service.search(query, from_timestamp)
        
        if not response or not response.tweets:
            logger.info(f"No tweets found for project {project.name} with query {query}")
            return True
            
        logger.error(f"Found {len(response.tweets)} tweets for project {project.name}")
        
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
                
                # если пост недостаточно популярен, то пропускаем
                post_shema = schemas.Post(
                    text="",
                    created_timestamp=0,
                    full_post_json={},
                    account_name="",
                    account_projects_statuses=[],
                    stats=schemas.PostStats(
                        favorite_count=main_post_data.favorite_count,
                        retweet_count=main_post_data.retweet_count,
                        reply_count=main_post_data.reply_count,
                        views_count=views
                    )
                ) # только для расчета показателя вовлеченности
                post_engagement_score = calculate_post_engagement_score(post_shema)
                logger.error(f"Post engagement:\n score: {post_engagement_score}\n post: {main_post_data.full_text} \n stats: {post_shema.stats}")
                # TODO: включить позже
                if post_engagement_score < settings.POST_SYNC_MINIMAL_ENGAGEMENT_SCORES and False:
                    logger.error(f"Skipping tweet due to low engagement score: {main_post_data.full_text}")
                    # пропускаем пост, не запоминаем его автора и статистику
                    continue
                
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
                tweets_added_count += 1
            except Exception as e:
                logger.error(f"Error processing tweet: {str(e)}")
                logger.error(f"Tweet data: {tweet}")
                continue

    except Exception as e:
        logger.error(f"Error fetching tweets: {str(e)}")
        logger.error(f"Project: {project.name}, Query: {query}")
        return False

    logger.error(f"Tweets added: {tweets_added_count}")
    return True


def calculate_post_engagement_score(post: schemas.Post) -> float:
    """
    Расчет показателя вовлеченности поста.
    """
    return post.stats.views_count*0.001 + post.stats.favorite_count * 1 + post.stats.retweet_count * 3 + post.stats.reply_count * 4

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
    project_feed.sort(key=lambda x: calculate_post_engagement_score(x), reverse=True)
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
        for p in posts[:limit if limit else len(posts)]:
            text = p.text[:max(len(p.text), max_len)]
            result += f"Post from {p.account_name}:\n{text}\n\n"
        return result

    project_feed_str = _prepare_posts(generation_settings.project_feed)

    # 2. Prompt (TODO: вынести в PromptService)
    SYSTEM_PROMPT = """
You are a social media content creator AI agent.
Your task it to analyze the project feed(in X social network) and generate unique and engaging tweets about the project like a human.
Important rules:
- Each tweet should be ≤280 characters, without numbering, quotes, and additional explanations.
- Return the result strictly in JSON format according to the tool schema, without any additional text.
- Analyze the project's feed and provide an up-to-date feed.
- Write in first or third person, as if you are a regular user.
- Your tweets be similar to the other tweets in the project feed.
- Do not mention the project directly (the system will do it) and do not mention anyone.
- Your tweets must be unique and not repeat one another.
- Use english language.

What can you write about?
- current events in the project
- coverage of discussed topics
- wishes, suggestions and thoughts about the project
- forecasts, assessments and other reflections

What should you not write about?
- answers to tweets of other users
- about the fact that you are an AI agent and your posts are based on the feed
- about the fact that you are using special tools for generating posts
- unfounded statements and false facts
    """

    USER_PROMPT = f"""# Project feed
{project_feed_str}
# End of project feed
IMPORTANT: follow all rules specified in the system prompt.
"""

    #logger.info(f"Generate posts examples prompt: \n{SYSTEM_PROMPT}")

    # 3. Response schema
    class GeneratePostsResponse(BaseModel):
        posts: List[str] = Field(description=f"List of {count} tweets, each ≤280 characters")

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
        max_tokens=512,
    ).with_structured_output(GeneratePostsResponse)

    try:
        result: GeneratePostsResponse = await llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ])
    except ValidationError as e:
        # Попытка ретрая с более жёстким системным промптом
        strict_prompt = SYSTEM_PROMPT + "\nВНИМАНИЕ: твой ответ ДОЛЖЕН быть JSON, полностью соответствовать схеме, иначе ошибка."
        result: GeneratePostsResponse = await llm.ainvoke([
            {"role": "system", "content": strict_prompt},
            {"role": "user", "content": USER_PROMPT},
        ])
    # 5. Пост-обработка и финальная валидация длины твитов
    cleaned: List[str] = []
    for tw in result.posts:
        tw_clean = tw.strip().replace("\n", " ")
        if len(tw_clean) > 280:
            tw_clean = tw_clean[:277] + "…"
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
4. Each variant must not exceed 280 characters.
5. Each variant must be written in the SAME LANGUAGE as the ORIGINAL TWEET.
6. Return the result strictly in JSON format according to the tool schema.
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



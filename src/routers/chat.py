from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc
import logging
import json
import jsonschema
from jsonschema import validate, ValidationError
import time

from src import schemas, enums, models, crud, utils, utils_base, exceptions
from src.core.middleware import get_current_user
from src.database import get_session
from src.services import user_context_service
from src.config.settings import settings
from src.config.mcp_servers import get_toolboxes, mcp_servers
from src.services.prompt_service import PromptService
from src.config import ai_models, subscription_plans

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatsResponse(BaseModel):
    chats: List[schemas.ChatSummary]

class ChatRequest(BaseModel):
    id: int
    from_nonce: int

class ChatResponse(BaseModel):
    chat: schemas.Chat

class AllSettingsResponse(BaseModel):
    default_chat_model_id: enums.Model
    chat_models: List[schemas.AiModelRestricted]
    chat_styles: List[enums.ChatStyle]
    chat_details_levels: List[enums.ChatDetailsLevel]

class UserMessageCreateRequest(BaseModel):
    chat_id: Optional[int] = None
    nonce: Optional[int] = None
    message: str
    model: Optional[enums.Model] = None
    chat_style: Optional[enums.ChatStyle] = None
    chat_details_level: Optional[enums.ChatDetailsLevel] = None
    selected_address: Optional[str] = None

class CreateMessageResponse(BaseModel):
    chat: schemas.Chat
    answer_message: schemas.MessageBase
    
class RegenerateMessageRequest(BaseModel):
    chat_id: int
    nonce: int
    model: Optional[enums.Model] = None
    chat_style: Optional[enums.ChatStyle] = None
    chat_details_level: Optional[enums.ChatDetailsLevel] = None
    selected_address: Optional[str] = None

class ChatDeleteRequest(BaseModel):
    chat_id: int

class DeleteResponse(BaseModel):
    success: bool = True


async def user_to_assistant_generate_data(user: models.User, create_message_request: UserMessageCreateRequest, chat: schemas.Chat) -> Tuple[schemas.AssistantGenerateData, schemas.ChatMessage]:
    model = create_message_request.model or settings.DEFAULT_AI_MODEL_ID
    next_nonce = create_message_request.nonce if create_message_request.nonce else (max(chat.messages.keys()) if chat.messages else 0)
    user_new_message = schemas.ChatMessage(
        sender=enums.Role.USER,
        recipient=enums.Role.ASSISTANT,
        nonce=next_nonce,
        content=schemas.MessageContent(
            message=create_message_request.message,
            settings=schemas.MessageGenerationSettings(
                model=model,
                chat_style=create_message_request.chat_style,
                chat_details_level=create_message_request.chat_details_level,
            )
        )
    )
    chat_settings = schemas.GenerateMessageSettings(
        model=model,
        chat_style=create_message_request.chat_style,
        chat_details_level=create_message_request.chat_details_level,
    )
    user_facts = [schemas.FactAboutUser(
        id=fact.id,
        description=fact.description,
        created_at=fact.created_at,
    ) for fact in user.facts]
    user_addresses:List[str] = [address.address for address in user.wallet_addresses]
    user_profile_data = schemas.UserProfileData(
        preferred_name=user.preferred_name,
        user_context=user.user_context,
        facts=user_facts,
        addresses=user_addresses,
        selected_address=create_message_request.selected_address,
    )
    assistant_generate_data = schemas.AssistantGenerateData(
        user=user_profile_data,
        chat=chat,
        chat_settings=chat_settings,
    )
    assistant_generate_data.chat.messages.update({next_nonce: [user_new_message]})
    return assistant_generate_data, user_new_message


@router.get("/all", response_model=ChatsResponse)
async def get_chats(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    chats = await crud.get_user_chats_summary(db, user.id)
    return ChatsResponse(chats=chats)


@router.get("/{id}", response_model=ChatResponse)
async def get_chat(id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    chat = await crud.get_user_chat(db, id, user.id, from_nonce=0)
    return ChatResponse(chat=chat)


@router.post("/delete", response_model=DeleteResponse)
async def delete_chat(request: ChatDeleteRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    await crud.delete_chat(db, request.chat_id, user.id)
    return DeleteResponse()


@router.get("/settings/all", response_model=AllSettingsResponse)
async def get_all_chat_settings():
    # TODO: добавить выключение моделей и сервисов
    # TODO: добавить кеширование данного ответа
    all_models: List[schemas.AiModelRestricted] = []
    for model in ai_models.all_ai_models:
        # ищем модель в планах
        from_plan_id = None
        for plan in subscription_plans.subscription_plans:
            if model.id in plan.available_models_ids:
                from_plan_id = plan.id
                break
        
        # если модель не найдена в планах, то она недоступна
        if from_plan_id is not None:
            all_models.append(schemas.AiModelRestricted(
                **model.model_dump(),
                from_plan_id=from_plan_id,
            ))
    
    return AllSettingsResponse(
        default_chat_model_id=settings.DEFAULT_AI_MODEL_ID,
        chat_models=all_models,
        chat_styles=list(enums.ChatStyle),
        chat_details_levels=list(enums.ChatDetailsLevel),
    )


async def stream_and_collect_messages(
    db: Session,
    event_generator: AsyncGenerator[str, None],
    user_message: schemas.ChatMessage,
    chat_id: int,
    user_id: int,
    delete_old_messages: bool = False,
):
    """
    ретранслирует события от генератора и собирает сообщения в список, чтобы сохранить их в базу
    """
    new_messages: List[schemas.MessageUnion] = [user_message]
    next_nonce = user_message.nonce + 1
    
    try:
        async for sse_event in event_generator:
            try:
                data = json.loads(sse_event[5:])
                event_type = data.get("type")
                # Собираем tool-calls
                if event_type == "tool_call":
                    pass
                elif event_type == "tool_result":
                    # Используем время из события, если оно есть
                    execution_time_ms = data.get("generation_time_ms", 0)
                    output = data.get("output")
                    logger.info(f"Тип output: {type(output)}, значение: {output}")
                    if isinstance(output, str):
                        try:
                            output = json.loads(output)
                        except Exception as e:
                            logger.error(f"Не удалось распарсить output как JSON: {output} ({e})")
                            output = {"raw": output}
                    if isinstance(output, list):
                        output = {"results": output}
                    
                    new_messages.append(schemas.ToolCallMessage(
                        sender=enums.Role.ASSISTANT,
                        recipient=enums.Role.USER,
                        nonce=next_nonce,
                        content=schemas.ToolCallContent(
                            name=data.get("name"),
                            input=data.get("args"),
                            output=output,
                        ),
                        generation_time_ms=execution_time_ms,
                    ))
                    next_nonce += 1
                elif event_type == "message_start":
                    pass
                elif event_type == "message_end":
                    # модель закончила отвечать
                    answer_text = data.get("text", "")
                    execution_time_ms = data.get("generation_time_ms")
                    # запоминаем сообщение, если оно не пустое
                    if answer_text:
                        new_messages.append(schemas.ChatMessage(
                            sender=enums.Role.ASSISTANT,
                            recipient=enums.Role.USER,
                            nonce=next_nonce,
                            generation_time_ms=execution_time_ms,
                            content=schemas.MessageContent(
                                message=answer_text,
                                settings=user_message.content.settings # те же настройки, что и у пользователя
                            )
                        ))
                        next_nonce += 1
            except Exception as e:
                logger.error(f"Ошибка при сборе сообщений: {e}", exc_info=True)
            finally:
                # в любом случае отправляем событие на клиент
                yield sse_event
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}", exc_info=True)
        error_message = f"Generation error: {str(e)}" if settings.DEBUG else "Generation error."
        new_messages.append(schemas.ChatMessage(
            sender=enums.Role.SYSTEM,
            recipient=enums.Role.USER,
            nonce=next_nonce,
            content=schemas.MessageContent(
                message=error_message,
                settings=user_message.content.settings # те же настройки, что и у пользователя
            )
        ))
        try:
            yield utils.to_sse("message_start", {"id": next_nonce})
            yield utils.to_sse("message_chunk", {"id": next_nonce, "text": error_message})
            yield utils.to_sse("message_end", {"id": next_nonce, "text": error_message, "generation_time_ms": 0})
            yield utils.to_sse("generation_end", {})
        except Exception as e:
            logger.error(f"Ошибка при отправке события: {e}", exc_info=True)
            # если не удалось отправить событие, то не сохраняем сообщения в и сразу выходим
            return

    
    logger.info(f"Сохраняем сообщения в базу: {new_messages}")
    try:
        # удаляем все сообщения после next_nonce, потому что если они есть, значит пользователь откатывается назад
        if delete_old_messages:
            await crud.delete_chat_messages(db, chat_id, user_message.nonce+1)
        await crud.add_chat_messages(db, chat_id, new_messages, user_id)
        
        # Запоминаем, что пользователь использовал кредит
        await crud.change_user_credits(db, user_id, 1)
    except Exception as e:
        logger.error(f"Ошибка сохранения сообщений: {e}", exc_info=True)

    return


@router.post("/message/stream", response_model=CreateMessageResponse)
async def create_message_stream(
    create_message_request: UserMessageCreateRequest,
    background_tasks: BackgroundTasks,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    # получаем подписку пользователя
    start_time = time.time()
    user_plan: enums.SubscriptionPlanType = await utils.check_user_subscription(user, db)
    user_subscription: schemas.SubscriptionPlanExtended = subscription_plans.get_subscription_plan(user_plan)
    logger.info(f"User plan: {user_plan}, time: {time.time() - start_time}")
    # получаем доступные модели для пользователя
    available_models = [model for model in user_subscription.available_models_ids]
    # если модель не указана, то разрешаем использовать дефолтную модель
    if create_message_request.model:
        # проверяем, что модель доступна для пользователя
        if create_message_request.model not in available_models:
            raise exceptions.APIError(code="unavailable_model", message="Model not available for your subscription plan.", status_code=403)
    
    # Проверяем, что у пользователь не использовал все кредиты
    if user.used_credits_today >= user_subscription.max_credits:
        raise exceptions.APIError(code="out_of_credits", message="Out of credits. Try again tomorrow.", status_code=403)
    
    # Если это новый чат 
    if create_message_request.chat_id is None:
        # запускаем фоновую задачу обработки контекста пользователя
        background_tasks.add_task(
            user_context_service.update_user_information,
            user_id=user.id
        )
        # создаём новый чат
        try:
            title = create_message_request.message.strip()[:30] + "..." if len(create_message_request.message) > 30 else create_message_request.message
            chat: schemas.Chat = await crud.create_chat(db, user.id, title)
            create_message_request.chat_id = chat.id
        except Exception as e:
            logger.error(f"Ошибка создания чата: {e}", exc_info=True)
            raise exceptions.APIError(code="chat_creation_failed", message="Failed to create new chat", status_code=500)
    else:
        # если чат существует, то получаем все сообщения до данного юзером nonce
        # если nonce указан, то получаем все сообщения до данного nonce
        try:
            if create_message_request.nonce is not None:
                chat: schemas.Chat = await crud.get_user_chat(db, create_message_request.chat_id, user.id, to_nonce=create_message_request.nonce-1)
            else:
                chat: schemas.Chat = await crud.get_user_chat(db, create_message_request.chat_id, user.id)
        except Exception as e:
            logger.error(f"Ошибка получения чата: {e}", exc_info=True)
            raise exceptions.APIError(code="chat_retrieval_failed", message="Failed to retrieve chat history", status_code=500)

    logger.info(f"Чат: {chat}")
    logger.info(f"Сообщения в чате: {chat.messages}")

    # собираем все данные для генерации ответа
    assistant_generate_data, user_message = await user_to_assistant_generate_data(user, create_message_request, chat)
    # получаем доступные ид тулбоксов для пользователя
    assistant_generate_data.toolbox_ids = [toolbox_id for toolbox_id in user_subscription.available_toolboxes_ids]
    prompt_service = PromptService(assistant_generate_data)
    
    # генерируем ответ от ИИ
    event_generator = utils.generate_ai_response_asstream(prompt_service)
    
    need_to_delete_old_messages = create_message_request.nonce is not None
    # Оборачиваем стриминг
    async def wrapped_stream():
        async for event in stream_and_collect_messages(
            db,
            event_generator,
            user_message,
            chat_id=create_message_request.chat_id,
            user_id=user.id,
            delete_old_messages=need_to_delete_old_messages,
        ):
            yield event
    
    return StreamingResponse(wrapped_stream(), media_type="text/event-stream")


class ToolsResponse(BaseModel):
    toolboxes: List[schemas.ToolboxRestricted]
    
    
@router.get("/tools/standart/all", response_model=ToolsResponse)
async def get_tools():
    # Сейчас каждый инстанс сервера будет иметь свой набор и получать его отдельно
    # ПРостое кеширование не избавит от лишних запросов при старте инстанса
    response: List[schemas.ToolboxRestricted] = []
    toolboxes = await get_toolboxes()
    for toolbox in toolboxes:
        from_plan_id = None
        for plan in subscription_plans.subscription_plans:
            if toolbox.id in plan.available_toolboxes_ids:
                from_plan_id = plan.id
                break
        if from_plan_id is not None:
            response.append(schemas.ToolboxRestricted(
                id=toolbox.id,
                name=toolbox.name,
                description=toolbox.description,
                tools=toolbox.tools,
                type=toolbox.type,
                from_plan_id=from_plan_id,
            ))
    
    return ToolsResponse(toolboxes=response)


class CallToolRequest(BaseModel):
    chat_id: Optional[int] = None
    toolbox_name: str
    tool_name: str
    input: Dict[str, Any]
    
class CallToolResponse(BaseModel):
    result: schemas.ToolCallMessage
    chat: schemas.Chat
    

@router.post("/tools/call", response_model=CallToolResponse)
async def call_tool(
    request: CallToolRequest,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    # Получаем доступные тулбоксы
    toolboxes = await get_toolboxes()
    
    # Валидируем наличие тулбокса
    toolbox = next((tb for tb in toolboxes if tb.name == request.toolbox_name), None)
    if not toolbox:
        raise exceptions.APIError(code="toolbox_not_found", message=f"Toolbox {request.toolbox_name} not found", status_code=404)
    
    # Валидируем наличие тула в тулбоксе
    tool = next((tool for tool in toolbox.tools if tool.name == request.tool_name), None)
    if not tool:
        raise exceptions.APIError(code="tool_not_found", message=f"Tool {request.tool_name} not found in toolbox {request.toolbox_name}", status_code=404)
    
    # Валидируем входные параметры на основе схемы
    try:
        validate_tool_input(tool.args_schema, request.input)
    except ValidationError as e:
        logger.error(f"Validation error for tool {request.tool_name}: {str(e)}")
        raise exceptions.APIError(code="invalid_input_parameters", message=f"Invalid input parameters: {str(e)}", status_code=400)
    
    # Инициализируем клиент MCP
    from src.services.mcp_client_service import MCPClient
    logger.info(f"Initializing MCP client for {request.toolbox_name}")

    
    # Вызываем тул
    try:
        logger.info(f"Calling tool {request.tool_name} with input {request.input}")
        start_time = time.time()
        
        # если это дефолтный тулбокс, то вызываем его напрямую
        if toolbox.type == enums.ToolboxType.DEFAULT:
            if hasattr(tool, "ainvoke"):
                # передаём ВСЮ dict одним позиционным аргументом
                result = await tool.ainvoke(request.input)
            else:
                result = tool.invoke(request.input)
            logger.info(f"Result1: {result}")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    result = {"raw": result}
                    
            if not isinstance(result, dict):
                result = {"raw": result}
            logger.info(f"Result2: {result}")
            
        else:
            # если это не дефолтный тулбокс, то вызываем его через MCP
                # Получаем сервер для выбранного тулбокса
            server = None 
            for server in mcp_servers.values():
                if server.name == request.toolbox_name:
                    server = server
                    break
            logger.info(f"Server: {server} {request.toolbox_name} {mcp_servers}")
            if not server:
                raise exceptions.APIError(code="server_configuration_not_found", message=f"Server configuration for {request.toolbox_name} not found", status_code=500)
            mcp_client = MCPClient(server)
            result: Dict[str, Any] = await mcp_client.invoke_tool(request.tool_name, request.input)
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Tool {request.tool_name} called successfully in {execution_time_ms} ms with result {result}")
    except Exception as e:
        logger.error(f"Error calling tool {request.tool_name}: {str(e)}", exc_info=True)
        raise exceptions.APIError(code="tool_call_failed", message=f"Error calling tool: {str(e)}", status_code=500)

    # Пытаемся сохранить сообщение в базу
    try:
        if request.chat_id is None:
            # создаём новый чат
            title = f"Call {request.tool_name} tool ..."
            chat = await crud.create_chat(db, user.id, title)
            request.chat_id = chat.id
            next_nonce = 0
        else:
            # получаем следующий nonce
            chat = await crud.get_user_chat(db, request.chat_id, user.id)
            logger.info(f"Chat: {chat}")
            if chat.messages:
                max_nonce = max(chat.messages.keys())
                next_nonce = max_nonce + 1
            else:
                next_nonce = 0
        

        tool_message = schemas.ToolCallMessage(
            content=schemas.ToolCallContent(
                name=request.tool_name,
                input=request.input,
                output=result
            ),
            nonce=next_nonce,
            sender=enums.Role.USER,
            recipient=enums.Role.USER,
            generation_time_ms=execution_time_ms,
        )
        await crud.add_chat_messages(db, request.chat_id, [tool_message], user.id)
    except exceptions.BusinessError as e:
        raise exceptions.APIError(code=e.code, message=e.message, status_code=400)
    except Exception as e:
        logger.error(f"Error adding message: {str(e)}", exc_info=True)
        raise exceptions.APIError(code="message_save_failed", message=f"Error saving message: {str(e)}", status_code=500)

    return CallToolResponse(result=tool_message, chat=chat)


def validate_tool_input(schema, input_data):
    """
    Валидирует входные параметры инструмента на основе его JSON-схемы.
    Конвертирует типы данных в соответствии со схемой.
    
    Args:
        schema (dict): JSON-схема инструмента
        input_data (dict): Входные параметры
        
    Raises:
        ValidationError: если параметры не соответствуют схеме
    """
    # Создаём копию входных данных для конвертации
    converted_data = input_data.copy()
    
    # Проверяем, что все обязательные параметры присутствуют
    if "required" in schema:
        for param in schema["required"]:
            if param not in input_data:
                raise ValidationError(f"Missing required parameter: {param}")
    
    # Конвертируем типы в соответствии со схемой
    if "properties" in schema:
        for param_name, param_data in input_data.items():
            if param_name in schema["properties"]:
                param_schema = schema["properties"][param_name]
                if "type" in param_schema:
                    if param_schema["type"] == "number" or param_schema["type"] == "integer":
                        try:
                            # Преобразуем строку в число если нужно
                            if isinstance(param_data, str):
                                if param_schema["type"] == "integer":
                                    converted_data[param_name] = int(param_data)
                                else:
                                    converted_data[param_name] = float(param_data)
                        except ValueError:
                            raise ValidationError(f"Parameter {param_name} must be a {param_schema['type']}")
                    elif param_schema["type"] == "boolean" and isinstance(param_data, str):
                        # Преобразуем строковые 'true'/'false' в boolean
                        if param_data.lower() == 'true':
                            converted_data[param_name] = True
                        elif param_data.lower() == 'false':
                            converted_data[param_name] = False
    
    # Проверяем дополнительные свойства
    if schema.get("additionalProperties") is False:
        for param in input_data:
            if param not in schema.get("properties", {}):
                raise ValidationError(f"Unknown parameter: {param}")
    
    # Выполняем полную валидацию схемы с преобразованными данными
    validate(instance=converted_data, schema=schema)
    
    # Обновляем исходные данные преобразованными значениями
    input_data.clear()
    input_data.update(converted_data)
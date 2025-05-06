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
from src.core.middleware import get_current_user, check_balance_and_update_token
from src.database import get_session
from src.config import settings
from src.services import user_context_service



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

class ProvidersResponse(BaseModel):
    models: List[enums.Model]

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


async def user_to_assistant_generate_data(user: models.User, create_message_request: UserMessageCreateRequest, chat: schemas.Chat, db: Session) -> Tuple[schemas.AssistantGenerateData, schemas.ChatMessage]:
    model = create_message_request.model or settings.DEFAULT_AI_MODEL
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

@router.get("/providers", response_model=ProvidersResponse)
async def get_providers():
    # TODO: добавить выключение моделей и сервисов
    # TODO: добавить кеширование данного ответа
    return ProvidersResponse(models=list(enums.Model))

@router.get("/all", response_model=ChatsResponse)
async def get_chats(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    chats = await crud.get_user_chats_summary(db, user.id)
    return ChatsResponse(chats=chats)

@router.get("/{id}", response_model=ChatResponse)
async def get_chat(id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    chat = await crud.get_user_chat(db, id, user.id, from_nonce=0)
    return ChatResponse(chat=chat)


async def stream_and_collect_messages(
    event_generator: AsyncGenerator[str, None],
    user_message: schemas.ChatMessage,
    chat_id: int,
    user_id: int,
    db: Session
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
                logger.info(f"Событие: {event_type} {data}")
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
        await crud.add_chat_messages(db, chat_id, new_messages, user_id)
    except Exception as e:
        logger.error(f"Ошибка сохранения сообщений: {e}", exc_info=True)

    return

@router.post("/message/regenerate", response_model=CreateMessageResponse)
async def regenerate_message(
    request: RegenerateMessageRequest,
    user: models.User = Depends(get_current_user),
    available_balance: bool = Depends(check_balance_and_update_token),
    db: Session = Depends(get_session),
):
    raise HTTPException(status_code=501, detail="This feature temporarily disabled")


@router.post("/message/stream", response_model=CreateMessageResponse)
async def create_message_stream(
    create_message_request: UserMessageCreateRequest,
    background_tasks: BackgroundTasks,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    from src.services.prompt_service import PromptService
    from src.mcp_servers import mcp_servers as mcp_servers_list
    from langchain_mcp_adapters.client import MultiServerMCPClient
    
    # Пытаемся списать кредиты
    try:
        # Обновляем баланс кредитов перед началом генерации
        await crud.refresh_user_credits(db, user)
        await crud.change_user_credits(db, user, -1)
    except exceptions.BusinessError as e:
        raise exceptions.APIError(code=e.code, message=e.message, status_code=403)
    except Exception as e:
        logger.error(f"Ошибка списания кредитов: {e}", exc_info=True)
        raise exceptions.APIError(code="credit_deduction_failed", message="Out of credits. Try again later.", status_code=500)
    
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
            if create_message_request.nonce:
                chat: schemas.Chat = await crud.get_user_chat(db, create_message_request.chat_id, user.id, to_nonce=create_message_request.nonce-1)
            else:
                chat: schemas.Chat = await crud.get_user_chat(db, create_message_request.chat_id, user.id)
        except Exception as e:
            logger.error(f"Ошибка получения чата: {e}", exc_info=True)
            raise exceptions.APIError(code="chat_retrieval_failed", message="Failed to retrieve chat history", status_code=500)

    # собираем все данные для генерации ответа
    assistant_generate_data, user_message = await user_to_assistant_generate_data(user, create_message_request, chat, db)
    prompt_service = PromptService(assistant_generate_data)
    
    # генерируем ответ от ИИ
    event_generator = utils.generate_ai_response_asstream(prompt_service)
    
    # Оборачиваем стриминг
    async def wrapped_stream():
        async for event in stream_and_collect_messages(
            event_generator,
            user_message,
            chat_id=create_message_request.chat_id,
            user_id=user.id,
            db=db,
        ):
            yield event
    
    return StreamingResponse(wrapped_stream(), media_type="text/event-stream")

@router.post("/delete", response_model=DeleteResponse)
async def delete_chat(request: ChatDeleteRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    await crud.delete_chat(db, request.chat_id, user.id)
    return DeleteResponse()


class ToolsResponse(BaseModel):
    toolboxes: List[schemas.Toolbox]
    
@router.get("/providers", response_model=ProvidersResponse)
async def get_providers():
    # TODO: добавить выключение моделей и сервисов
    # TODO: добавить кеширование данного ответа
    return ProvidersResponse(models=list(enums.Model))

@router.get("/tools/standart/all", response_model=ToolsResponse)
async def get_tools():
    # Сейчас каждый инстанс сервера будет иметь свой набор и получать его отдельно
    # ПРостое кеширование не избавит от лишних запросов при старте инстанса
    from src.mcp_servers import get_toolboxes
    result = await get_toolboxes()
    return ToolsResponse(toolboxes=result)

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
    from src.mcp_servers import get_toolboxes
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
        validate_tool_input(tool.inputSchema, request.input)
    except ValidationError as e:
        logger.error(f"Validation error for tool {request.tool_name}: {str(e)}")
        raise exceptions.APIError(code="invalid_input_parameters", message=f"Invalid input parameters: {str(e)}", status_code=400)
    
    # Получаем сервер для выбранного тулбокса
    from src.mcp_servers import mcp_servers
    server = next((server for server in mcp_servers if server.name == request.toolbox_name), None)
    if not server:
        raise exceptions.APIError(code="server_configuration_not_found", message=f"Server configuration for {request.toolbox_name} not found", status_code=500)
    
    # Инициализируем клиент MCP
    from src.services.mcp_client_service import MCPClient
    logger.info(f"Initializing MCP client for {request.toolbox_name}")
    mcp_client = MCPClient(server)
    
    # Вызываем тул
    try:
        logger.info(f"Calling tool {request.tool_name} with input {request.input}")
        start_time = time.time()
        
        result = await mcp_client.invoke_tool(request.tool_name, request.input)
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Tool {request.tool_name} called successfully in {execution_time_ms} ms")
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
            next_nonce = await crud.get_next_nonce(db, request.chat_id, user.id)
        
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
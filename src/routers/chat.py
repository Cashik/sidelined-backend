from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc

from src import schemas, enums, models, crud, utils
from src.core.middleware import get_current_user, check_balance_and_update_token
from src.database import get_session
from src.config import settings
from src.exceptions import MessageNotFoundException, InvalidMessageTypeException
from src.services import user_context_service

import logging
import json
import jsonschema
from jsonschema import validate, ValidationError

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

class CreateMessageResponse(BaseModel):
    chat: schemas.Chat
    answer_message: schemas.Message
    
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


async def user_to_assistant_generate_data(user: models.User, create_message_request: schemas.MessageCreate, db: Session) -> schemas.AssistantGenerateData:
    model = create_message_request.model or settings.DEFAULT_AI_MODEL
    
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
    chat = await crud.get_user_chat(db, create_message_request.chat_id, user.id)
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
    return assistant_generate_data

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
    # В режиме DEBUG возвращаем все сообщения, иначе только для пользователя
    recipient = None if settings.DEBUG else enums.Role.USER
    chat = await crud.get_user_chat(db, id, user.id, from_nonce=0, recipient=recipient)
    return ChatResponse(chat=chat)

@router.post("/message", response_model=CreateMessageResponse)
async def create_message(
    create_message_request: schemas.MessageCreate,
    background_tasks: BackgroundTasks,
    user: models.User = Depends(get_current_user),
    available_balance: bool = Depends(check_balance_and_update_token),
    db: Session = Depends(get_session),
):
    # добавляем сообщение пользователя в чат
    # если nonce не указан, то добавляем в конец чата
    # !если nonce указан, то последующие сообщения удаляются
    # TODO: стоит ли доверять nonce с клиента? если нет, то нужно отделить возможность возвращаться к старым сообщениям в отдельный метод
    
    # Используем модель по умолчанию, если не задана
    model = create_message_request.model or settings.DEFAULT_AI_MODEL
    
    user_message = schemas.Message(
        content=create_message_request.message,
        sender=enums.Role.USER,
        recipient=enums.Role.ASSISTANT,
        model=model,
        nonce=create_message_request.nonce,
        chat_style=create_message_request.chat_style,
        chat_details_level=create_message_request.chat_details_level,
    )
    
    # Если это новый чат (был только что создан), 
    is_new_chat = create_message_request.chat_id is None
    if is_new_chat:
        # запускаем фоновую задачу обработки контекста пользователя
        background_tasks.add_task(
            user_context_service.update_user_information,
            user_id=user.id
        )
        # создаём новый чат
        title = user_message.content.strip()[:30] + "..." if len(user_message.content) > 30 else user_message.content
        chat = await crud.create_chat(db, user.id, title)
        create_message_request.chat_id = chat.id

    # генерируем ответ от ИИ
    assistant_generate_data: schemas.AssistantGenerateData = await user_to_assistant_generate_data(user, create_message_request, db)
    assistant_generate_data.chat.messages.update({user_message.nonce: [user_message]})
    answer_messages: List[schemas.Message] = await utils.get_ai_answer(assistant_generate_data, user.id, db)
    
    # добавляем новые сообщения в чат
    chat: schemas.Chat = await crud.add_message(db, create_message_request.chat_id, user_message, user.id)
    for message in answer_messages:
        chat: schemas.Chat = await crud.add_message(db, chat.id, message, user.id)
    return CreateMessageResponse(chat=chat, answer_message=answer_messages[0])

@router.post("/message/regenerate", response_model=CreateMessageResponse)
async def regenerate_message(
    request: RegenerateMessageRequest,
    user: models.User = Depends(get_current_user),
    available_balance: bool = Depends(check_balance_and_update_token),
    db: Session = Depends(get_session),
):
    # получаем сообщение для регенерации
    message_to_regenerate = db.execute(
        select(models.Message)
        .where(
            models.Message.chat_id == request.chat_id,
            models.Message.nonce == request.nonce
        )
        .order_by(desc(models.Message.selected_at))
        .limit(1)
    ).scalar_one_or_none()
    
    if not message_to_regenerate:
        raise MessageNotFoundException()
        
    if message_to_regenerate.sender != enums.Role.ASSISTANT:
        raise InvalidMessageTypeException("Can only regenerate assistant messages")
    
    # получаем все сообщения до запрошенного
    chat: schemas.Chat = await crud.get_user_chat(db, request.chat_id, user.id, to_nonce=request.nonce-1)
    
    # Используем модель по умолчанию, если не задана
    model = request.model or settings.DEFAULT_AI_MODEL
    
    # генерируем новое сообщение от ИИ
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
        selected_address=request.selected_address,
    )
    chat_settings = schemas.GenerateMessageSettings(
        model=model,
        chat_style=request.chat_style,
        chat_details_level=request.chat_details_level,
    )
    assistant_generate_data = schemas.AssistantGenerateData(
        user=user_profile_data,
        chat=chat,
        chat_settings=chat_settings,
    )
    answer_messages: List[schemas.Message] = await utils.get_ai_answer(assistant_generate_data, user.id, db)
    # добавляем новые сообщения в чат
    for message in answer_messages:
        chat: schemas.Chat = await crud.add_message(db, chat.id, message, user.id)
    return CreateMessageResponse(chat=chat, answer_message=answer_messages[0])



@router.post("/message/stream", response_model=CreateMessageResponse)
async def test_longgraph(
    create_message_request: schemas.MessageCreate,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    from src.services.prompt_service import PromptService
    from src.mcp_servers import mcp_servers as mcp_servers_list
    from langchain_mcp_adapters.client import MultiServerMCPClient
    # Используем модель по умолчанию, если не задана
    model = create_message_request.model or settings.DEFAULT_AI_MODEL
    
        # Используем модель по умолчанию, если не задана
    model = create_message_request.model or settings.DEFAULT_AI_MODEL
    
    user_message = schemas.Message(
        content=create_message_request.message,
        sender=enums.Role.USER,
        recipient=enums.Role.ASSISTANT,
        model=model,
        nonce=create_message_request.nonce,
        chat_style=create_message_request.chat_style,
        chat_details_level=create_message_request.chat_details_level,
    )
    
    # Если это новый чат (был только что создан), 
    is_new_chat = create_message_request.chat_id is None
    if is_new_chat:
        # создаём новый чат
        title = user_message.content.strip()[:30] + "..." if len(user_message.content) > 30 else user_message.content
        chat = await crud.create_chat(db, user.id, title)
        create_message_request.chat_id = chat.id

    # генерируем ответ от ИИ
    assistant_generate_data: schemas.AssistantGenerateData = await user_to_assistant_generate_data(user, create_message_request, db)
    assistant_generate_data.chat.messages.update({user_message.nonce: [user_message]})
    
    prompt_service = PromptService(assistant_generate_data)
    answer_stream = utils.generate_ai_response_asstream(prompt_service)
    return StreamingResponse(answer_stream, media_type="text/event-stream")

@router.post("/delete", response_model=DeleteResponse)
async def delete_message(request: ChatDeleteRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    await crud.delete_chat(db, request.chat_id, user.id)
    return DeleteResponse()

@router.post("/test-background", response_model=Dict[str, str])
async def test_background(
    message: str,
    background_tasks: BackgroundTasks,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    """
    Тестовый эндпоинт для проверки работы BackgroundTasks.
    """
    # Запускаем тестовую фоновую задачу
    background_tasks.add_task(
        user_context_service.test_background_task,
        user_id=user.id,
        message=message
    )
    
    return {"status": "success", "message": "Тестовая фоновая задача запущена"}


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
    toolbox_name: str = "Thirdweb"
    tool_name: str
    input: Dict[str, Any]
    
class CallToolResponse(BaseModel):
    success: bool = True
    result: Dict[str, Any] | None = None
    as_message: str | None = None
    execution_time: float | None = None
    
@router.post("/tools/call", response_model=CallToolResponse)
async def call_tool(request: CallToolRequest):
    # Получаем доступные тулбоксы
    from src.mcp_servers import get_toolboxes
    toolboxes = await get_toolboxes()
    
    # Валидируем наличие тулбокса
    toolbox = next((tb for tb in toolboxes if tb.name == request.toolbox_name), None)
    if not toolbox:
        raise HTTPException(status_code=404, detail=f"Toolbox {request.toolbox_name} not found")
    
    # Валидируем наличие тула в тулбоксе
    tool = next((tool for tool in toolbox.tools if tool.name == request.tool_name), None)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool {request.tool_name} not found in toolbox {request.toolbox_name}")
    
    # Валидируем входные параметры на основе схемы
    try:
        validate_tool_input(tool.inputSchema, request.input)
    except ValidationError as e:
        logger.error(f"Validation error for tool {request.tool_name}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input parameters: {str(e)}")
    
    # Получаем сервер для выбранного тулбокса
    from src.mcp_servers import mcp_servers
    server = next((server for server in mcp_servers if server.name == request.toolbox_name), None)
    if not server:
        raise HTTPException(status_code=500, detail=f"Server configuration for {request.toolbox_name} not found")
    
    # Инициализируем клиент MCP
    from src.services.mcp_client_service import MCPClient
    logger.info(f"Initializing MCP client for {request.toolbox_name}")
    mcp_client = MCPClient(server)
    
    # Вызываем тул
    try:
        logger.info(f"Calling tool {request.tool_name} with input {request.input}")
        import time
        start_time = time.time()
        
        result = await mcp_client.invoke_tool(request.tool_name, request.input)
        
        execution_time = time.time() - start_time
        logger.info(f"Tool {request.tool_name} called successfully in {execution_time:.2f} seconds")
        
        return CallToolResponse(
            result=result,
            execution_time=execution_time
        )
    except Exception as e:
        logger.error(f"Error calling tool {request.tool_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calling tool: {str(e)}")

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
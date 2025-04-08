from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc

from src import schemas, enums, models, crud, utils
from src.core.middleware import get_current_user, check_balance_and_update_token
from src.database import get_session
from src.config import settings
from src.exceptions import MessageNotFoundException, InvalidMessageTypeException

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

class ChatDeleteRequest(BaseModel):
    chat_id: int

class DeleteResponse(BaseModel):
    success: bool = True


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

@router.post("/message", response_model=CreateMessageResponse)
async def create_message(
    create_message_request: schemas.MessageCreate,
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
        created_at=utils.now_timestamp(),
        selected_at=utils.now_timestamp(),
    )
    chat: schemas.Chat = await crud.add_message(db, create_message_request.chat_id, user_message, user.id)
    # генерируем ответ от ИИ
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
    user_profile_data = schemas.UserProfileData(
        preferred_name=user.preferred_name,
        user_context=user.user_context,
        facts=user_facts,
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
    user_profile_data = schemas.UserProfileData(
        preferred_name=user.preferred_name,
        user_context=user.user_context,
        facts=user_facts,
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

@router.post("/delete", response_model=DeleteResponse)
async def delete_message(request: ChatDeleteRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    await crud.delete_chat(db, request.chat_id, user.id)
    return DeleteResponse()

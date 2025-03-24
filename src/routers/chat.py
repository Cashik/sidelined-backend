from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlmodel import Session
from sqlalchemy import select, func

from src import schemas, enums, models, crud, utils
from src.core.middleware import get_current_user
from src.database import get_session

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
    services: List[enums.Service]

class CreateMessageResponse(BaseModel):
    chat: schemas.Chat
    answer_message: schemas.Message
    
class RegenerateMessageRequest(BaseModel):
    chat_id: int
    nonce: int
    model: enums.Model

class ChatDeleteRequest(BaseModel):
    chat_id: int

class DeleteResponse(BaseModel):
    success: bool = True


@router.get("/providers", response_model=ProvidersResponse)
async def get_providers():
    # TODO: добавить выключение моделей и сервисов
    # TODO: добавить кеширование данного ответа
    return ProvidersResponse(models=list(enums.Model), services=list(enums.Service))

@router.get("/all", response_model=ChatsResponse)
async def get_chats(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    chats = await crud.get_user_chats_summary(db, user.id)
    return ChatsResponse(chats=chats)

@router.get("/{id}", response_model=ChatResponse)
async def get_chat(id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    chat = await crud.get_user_chat(db, id, user.id, from_nonce=0)
    return ChatResponse(chat=chat)

@router.post("/message", response_model=CreateMessageResponse)
async def create_message(create_message_request: schemas.MessageCreate, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    # добавляем сообщение пользователя в чат
    # если nonce не указан, то добавляем в конец чата
    # !если nonce указан, то последующие сообщения удаляются
    # TODO: стоит ли доверять nonce с клиента? если нет, то нужно отделить возможность возвращаться к старым сообщениям в отдельный метод
    user_message = schemas.Message(
        content=create_message_request.message,
        sender=enums.Role.USER,
        recipient=enums.Role.ASSISTANT,
        model=create_message_request.model,
        nonce=create_message_request.nonce,
        created_at=utils.now_timestamp(),
        selected_at=utils.now_timestamp(),
    )
    chat: schemas.Chat = await crud.add_message(db, create_message_request.chat_id, user_message, user.id)
    # генерируем ответ от ИИ
    answer_message: schemas.Message = await utils.get_ai_answer(chat, create_message_request.model)
    # добавляем ответ в чат
    chat: schemas.Chat = await crud.add_message(db, chat.id, answer_message, user.id)
    return CreateMessageResponse(chat=chat, answer_message=answer_message)

@router.post("/message/regenerate", response_model=CreateMessageResponse)
async def regenerate_message(request: RegenerateMessageRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    # получаем все сообщения до запрошенного
    chat: schemas.Chat = await crud.get_user_chat(db, request.chat_id, user.id, to_nonce=request.nonce-1)
    # генерируем новое сообщение от ИИ
    answer_message: schemas.Message = await utils.get_ai_answer(chat, request.model)
    # добавляем новое сообщение в чат
    chat: schemas.Chat = await crud.add_message(db, chat.id, answer_message, user.id)
    return CreateMessageResponse(chat=chat, answer_message=answer_message)

@router.post("/delete", response_model=DeleteResponse)
async def delete_message(request: ChatDeleteRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    await crud.delete_chat(db, request.chat_id, user.id)
    return DeleteResponse()

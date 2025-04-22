from openai import OpenAI
import logging
from pydantic import BaseModel
from typing import List, Optional, Any

from src.providers.base import AIProvider
from src.config import settings
from src.services.prompt_service import PromptService
from src import schemas, enums

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# Определение функций для работы с заметками
tools = [
    {
        "type": "function",
        "function": {
            "name": "notes_add",
            "description": "Add new notes about the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_notes": {
                        "type": "array",
                        "description": "Each note should be short and concise.",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["new_notes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "notes_remove",
            "description": "Delete wrong or unactual notes from list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "description": "List of valid IDs from the list of notes to delete",
                        "items": {
                            "type": "integer"
                        }
                    }
                },
                "required": ["ids"]
            }
        }
    }
]

class OpenAIProvider(AIProvider):
    def __init__(self):
        pass

    async def generate_response(self, prompt_service: PromptService) -> schemas.GeneratedResponse:
        logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
        logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
        
        messages = []
        system_prompt = prompt_service.generate_system_prompt()
        messages.append(SystemMessage(content=system_prompt))
        
        # Добавление сообщений из истории чата
        for nonce in sorted(prompt_service.generate_data.chat.messages.keys()):
            message = max(prompt_service.generate_data.chat.messages[nonce], key=lambda x: x.selected_at)
            if message.sender is enums.Role.USER:
                messages.append(HumanMessage(content=message.content))
            else:
                messages.append(AIMessage(content=message.content))
        
        logger.info(f"Sending request to OpenAI with messages: {messages}")
        
        model = init_chat_model(
            model=prompt_service.generate_data.chat_settings.model.value,
            model_provider="openai",
            api_key=settings.OPENAI_API_KEY
        )
        
        response = model.invoke(messages)
        
        # Обработка ответа
        answer_text = response.content if response.content else ""
        
        result = schemas.GeneratedResponse(
            text=answer_text,
            function_calls=[]
        )
        
        logger.info(f"Answer: {result}")
        return result

    async def adapt_tools(self, tools: List[schemas.Tool]):
        return tools


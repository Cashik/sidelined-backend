from openai import OpenAI
import logging
from pydantic import BaseModel
from typing import List, Optional, Any

from src.providers.base import AIProvider
from src.config import settings
from src.services.prompt_service import PromptService
from src import schemas, enums

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
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_response(self, prompt_service: PromptService) -> schemas.GeneratedResponse:
        logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
        logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
        
        messages = []
        system_prompt = prompt_service.generate_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # Добавление сообщений из истории чата
        for nonce in sorted(prompt_service.generate_data.chat.messages.keys()):
            message = max(prompt_service.generate_data.chat.messages[nonce], key=lambda x: x.selected_at)
            messages.append({
                "role": message.sender.value,
                "content": message.content
            })
        
        messages.append({
            "role": "system",
            "content": "Do not forget to update the list of notes using your available functions if needed."
        })
        
        logger.info(f"Sending request to OpenAI with messages: {messages}")
        
        model_name = prompt_service.generate_data.chat_settings.model.value
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1500
        )
        
        # Обработка ответа
        message = response.choices[0].message
        answer_text = message.content if message.content else ""
        
        # Обработка вызовов функций
        function_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "notes_add":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    function_calls.append(schemas.AddFactsFunctionCall(args=args.get("new_notes")))
                elif tool_call.function.name == "notes_remove":
                    import json
                    args = json.loads(tool_call.function.arguments)
                    function_calls.append(schemas.RemoveFactsFunctionCall(args=args.get("ids")))
        
        if not answer_text:
            # модель не выдала ответ, но вызвала функции. Принуждаем ее сгенерировать ответ без тулзов.
            messages = messages[:-1] # удаляем последнее сообщение с инструкцией по обновлению списка заметок
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            message = response.choices[0].message
            answer_text = message.content if message.content else ""
        
        result = schemas.GeneratedResponse(
            text=answer_text,
            function_calls=function_calls
        )
        
        logger.info(f"Answer: {result}")
        return result

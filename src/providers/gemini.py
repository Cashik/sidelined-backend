from google import genai
from google.genai import types
import logging
from pydantic import BaseModel
from typing import List, Optional, Any
from src.providers.base import AIProvider
from src.config import settings
from src.services.prompt_service import PromptService
from src import enums, schemas

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

class GeminiNotesResponse(BaseModel):
    delete_notes: Optional[List[int]] = None
    new_notes: Optional[List[str]] = None


class GeminiProvider(AIProvider):
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
        
        logger.info(f"Sending request to Gemini with messages: {messages}")
        
        model = init_chat_model(
            model=prompt_service.generate_data.chat_settings.model.value,
            model_provider="google_genai",
            api_key=settings.GEMINI_API_KEY
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
        
    async def generate_notes_response(self, prompt: str) -> GeminiNotesResponse:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiNotesResponse
        )
        
        response = client.models.generate_content(
            model=enums.Model.GEMINI_2_5_PRO.value,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=generate_content_config
        )
        return response.parsed

    async def adapt_tools(self, tools: List[schemas.Tool]):
        return tools

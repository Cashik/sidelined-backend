from google import genai
from google.genai import types
import logging

from src.providers.base import AIProvider
from src.config import settings
from src.services.prompt_service import PromptService
from src import enums

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

class GeminiProvider(AIProvider):
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def generate_response(self, prompt_service: PromptService) -> str:
        logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
        logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
        
        contents = []
        system_prompt = prompt_service.generate_system_prompt()
        
        # Добавляем историю сообщений
        for nonce in sorted(prompt_service.generate_data.chat.messages.keys()):
            message = max(prompt_service.generate_data.chat.messages[nonce], key=lambda x: x.selected_at)
            message_text = message.content if message.content else "-"
            role = "model" if message.sender != enums.Role.USER else enums.Role.USER
            prefix = "system_hint:" if message.sender == enums.Role.SYSTEM else ""
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=prefix + message_text)]
            ))
        
        logger.info(f"Sending request to Gemini with contents: {contents}")
        
        model_name = prompt_service.generate_data.chat_settings.model.value
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.7,
            system_instruction=[
                types.Part.from_text(text=system_prompt)
            ]
        )
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config
        )
        
        return response.text

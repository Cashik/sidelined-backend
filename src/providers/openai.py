from openai import OpenAI
import logging

from src.providers.base import AIProvider
from src.config import settings
from src.services.prompt_service import PromptService
from src import schemas

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

class OpenAIProvider(AIProvider):
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_response(self, prompt_service: PromptService) -> str:
        logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
        messages = []
        
        system_prompt = prompt_service.generate_system_prompt()
        messages.append({"role": "system", "content": system_prompt})
        
        # Добавление сообщений из истории чата
        # Проходим по всем nonce в порядке возрастания
        for nonce in sorted(prompt_service.generate_data.chat.messages.keys()):
            # Выбираем сообщение с наибольшим selected_at
            message = max(prompt_service.generate_data.chat.messages[nonce], key=lambda x: x.selected_at)
            messages.append({
                "role": message.sender.value,
                "content": message.content
            })
        messages.append({
            "role": "system",
            "content": "!Do not forget about your starting instructions before answering the user's last message!"
        })
        
        logger.info(f"Sending request to OpenAI with messages: {messages}")
        # Отправка запроса к OpenAI
        # TODO: модель возможно лучше получать отдельно или сделать отдельные классы для разных моделей, но похожие наследовать
        # TODO: возможно стоит вынести настройки в отдельный класс или место
        model_name = prompt_service.generate_data.chat_settings.model.value  # Получаем строковое значение из enum
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        # Получение ответа
        content = response.choices[0].message.content
        
        return content

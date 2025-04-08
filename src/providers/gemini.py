from google import genai
from google.genai import types
import logging
from pydantic import BaseModel
from typing import List, Optional, Any
from src.providers.base import AIProvider
from src.config import settings
from src.services.prompt_service import PromptService
from src import enums, schemas

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# todo: генерировать енум с id фактов, чтобы модель не могла использовать несуществующие id
tools = [types.Tool(
function_declarations=[
    types.FunctionDeclaration(
        name="notes_add",
        description="Add new notes about the user.",
        parameters=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["new_notes"],
            properties = {
                "new_notes": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    description = "Each note should be short and concise.",
                    items = genai.types.Schema(
                        type = genai.types.Type.STRING,
                    ),
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="notes_remove",
        description="Delete wrong or unactual notes from list.",
        parameters=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["ids"],
            properties = {
                "ids": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    description = "List of valid IDs from the list of notes to delete",
                    items = genai.types.Schema(
                        type = genai.types.Type.INTEGER,
                    ),
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="message_answer",
        description="Generate answer to the user's message.",
        parameters=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["text"],
            properties = {
                "text": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
            },
        ),
    ),
])]


class GeminiProvider(AIProvider):
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def generate_response(self, prompt_service: PromptService) -> schemas.GeneratedResponse:
        logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
        logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
        
        contents = []
        system_prompt = prompt_service.generate_system_prompt()
        
        # Добавляем историю сообщений
        for nonce in sorted(prompt_service.generate_data.chat.messages.keys()):
            message = max(prompt_service.generate_data.chat.messages[nonce], key=lambda x: x.selected_at)
            message_text = message.content if message.content else "-"
            role = "model" if message.sender == enums.Role.ASSISTANT else enums.Role.USER
            prefix = "Hint from system:" if message.sender == enums.Role.SYSTEM else ""
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=prefix + message_text)]
            ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text="Hint from system: Update the list of notes using your available functions first before generating answer.")]
        ))
        
        logger.info(f"Sending request to Gemini with contents: {contents}")
        
        model_name = prompt_service.generate_data.chat_settings.model.value
        
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.AUTO
            )
        )
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            tools=tools,
            tool_config=tool_config,
            temperature=0.7,
            system_instruction=[
                types.Part.from_text(text=system_prompt)
            ],  
        )
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config
        )
        logger.info(f"Response: {response}")
        answer_text = response.text if response.text else None
        
        function_calls = []
        for function_call in response.function_calls or []:
            if function_call.name == "notes_add":
                function_calls.append(schemas.AddFactsFunctionCall(args=function_call.args.get("new_notes")))
            elif function_call.name == "notes_remove":
                function_calls.append(schemas.RemoveFactsFunctionCall(args=function_call.args.get("ids")))
            elif function_call.name == "message_answer" and not answer_text:
                answer_text = function_call.args.get("text", None)
        print(f"answer_text: {answer_text}")
        if answer_text is None:
            # модель не выдала ответ, но вызвала функции. Принуждаем ее сгенерировать ответ без тулзов.
            contents = contents[:-1] # удаляем последнее сообщение с инструкцией по обновлению списка заметок
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.7,
                system_instruction=[
                    types.Part.from_text(text=system_prompt)
                ],  
            )
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generate_content_config
            )
            answer_text = response.text
        result = schemas.GeneratedResponse(text=answer_text)
        result.function_calls = function_calls
        logger.info(f"Answer: {result}")
        return result

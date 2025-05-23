from google import genai
from google.genai import types
import logging
from pydantic import BaseModel
from typing import List, Optional, Any
from src.providers.base import AIProvider
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from src.config.settings import settings
from src.services.prompt_service import PromptService
from src import enums, schemas


logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

class GeminiNotesResponse(BaseModel):
    delete_notes: Optional[List[int]] = None
    new_notes: Optional[List[str]] = None


class GeminiProvider(AIProvider):
    def __init__(self):
        pass

    async def generate_response(self, prompt_service: PromptService, tools: List[schemas.Tool] = []) -> schemas.GeneratedResponse:
        logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
        logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
        
        avoid_system_role = prompt_service.generate_data.chat_settings.model.value in [enums.Model.GEMINI_2_FLASH.value]
        logger.info(f"Avoid system role: {avoid_system_role}")
        messages = prompt_service.generate_langchain_messages(avoid_system_role)
        

        logger.info(f"Sending request to Gemini with messages: {messages}")
        
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        llm = init_chat_model(
            model=prompt_service.generate_data.chat_settings.model.value,
            model_provider="google_genai",
            api_key=settings.GEMINI_API_KEY
        )
        

        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )
        
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=10, #TODO: add max_iterations
            verbose=settings.DEBUG
        )
        
        response = await executor.ainvoke({"chat_history": messages})
        
        # Обработка ответа
        answer_text = response["output"] if response["output"] else ""
        
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

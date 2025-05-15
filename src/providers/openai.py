import logging
from typing import List, Optional, Any
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from src.providers.base import AIProvider
from src.config.settings import settings
from src.services.prompt_service import PromptService
from src import schemas, enums



logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


class OpenAIProvider(AIProvider):
    def __init__(self):
        pass

    async def generate_response(self, prompt_service: PromptService, tools: List[schemas.Tool] = []) -> schemas.GeneratedResponse:
        logger.info(f"Generating response for prompt: {prompt_service.generate_data.chat.messages}")
        logger.info(f"Model: {prompt_service.generate_data.chat_settings.model}")
        
        messages = prompt_service.generate_langchain_messages()
        
        logger.info(f"Sending request to OpenAI with messages: {messages}")
        
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        llm = init_chat_model(
            model=prompt_service.generate_data.chat_settings.model.value,
            model_provider="openai",
            api_key=settings.OPENAI_API_KEY
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

    async def adapt_tools(self, tools: List[schemas.Tool]):
        return tools


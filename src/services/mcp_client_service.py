import os
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from mcp import ClientSession, types
from mcp.client.sse import sse_client

from src import schemas

logger = logging.getLogger(__name__)

class MCPClientService:

    def __init__(self, server: schemas.MCPSSEServer):
        self.name = server.name
        self.description = server.description
        self.url = server.url
        self.transport = server.transport
        self._tools: List[types.Tool] = []
        
        logger.info(f"Инициализация MCP клиента. {self.name} - {self.description} - {self.url}")
    
    async def initialize(self):
        """Инициализирует клиент и получает список инструментов"""
        async with sse_client(url=self.url) as (read, write):
            async with ClientSession(
                read, write
            ) as session:
                # Initialize the connection
                await session.initialize()

                # List available tools
                self._tools = (await session.list_tools()).tools

                print(f"Successfully initialized MCP client. Tools: {self._tools}")
    
    @asynccontextmanager
    async def get_session(self):
        """Предоставляет активную сессию MCP для работы с инструментами"""
        async with sse_client(url=self.url) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                logger.info(f"Successfully initialized MCP client session")
                yield session
                
    async def get_tools(self) -> List[types.Tool]:
        """Получить список доступных инструментов"""
        return self._tools
    
    async def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Вызвать инструмент с заданными параметрами"""
        async with self.get_session() as session:
            logger.info(f"Invoking tool {tool_name} with params {params}")
            result = await session.call_tool(tool_name, params)
            logger.info(f"Result of invoke tool {tool_name}: {result}")
            if result.isError:
                return {"text": f"Error occurred while invoking tool: {result.content[0].text}"}
            else:
                if result.content[0].type == 'text':
                    return {"text": result.content[0].text}
                else:
                    return {"text": "Unsupported content type received from MCP"}



import os
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from mcp import ClientSession, types
from mcp.client.sse import sse_client
from mcp.client.websocket import websocket_client
from src import schemas

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self, server: schemas.MCPServer):
        self.server = server

    @asynccontextmanager
    async def get_session(self):
        if self.server.transport == "sse":
            async with sse_client(url=self.server.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        elif self.server.transport == "websocket":
            async with websocket_client(url=self.server.url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        else:
            raise NotImplementedError(f"Transport {self.server.transport} is not implemented")
    
    async def get_tools(self) -> List[types.Tool]:
        async with self.get_session() as session:
            
            self._tools = (await session.list_tools()).tools
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

    




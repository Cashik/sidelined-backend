from typing import List
import smithery
from langchain_mcp_adapters.client import MultiServerMCPClient
import logging

from src import schemas
from src.config import settings
from src.services.mcp_client_service import MCPClient

logger = logging.getLogger(__name__)

mcp_servers = []
if settings.EXA_SEARCH_API_KEY and settings.SMITHERY_API_KEY and False:
    exa_search_url = smithery.create_smithery_url("wss://server.smithery.ai/exa/ws", {
        "exaApiKey": settings.EXA_SEARCH_API_KEY
    }) + f"&api_key={settings.SMITHERY_API_KEY}"
    mcp_servers.append(schemas.MCPWebSocketServer(
        name="Exa Search",
        description="Fast, intelligent web search and crawling.",
        url=exa_search_url
    ))

if settings.EVM_AGENT_KIT_SSE_URL:
    mcp_servers.append(schemas.MCPSSEServer(
        name="EVM Agent Kit",
        description="A lot of blockchain tools",
        url=settings.EVM_AGENT_KIT_SSE_URL
    ))

toolboxes: List[schemas.Toolbox] = []

async def sync_toolboxes():
    for server in mcp_servers:
        mcp_client = MCPClient(server)
        tools = await mcp_client.get_tools()
        toolbox = schemas.Toolbox(
            name=server.name,
            description=server.description,
            tools=tools
        )
        toolboxes.append(toolbox)
    logger.info(f"Синхронизировали инструменты для {len(mcp_servers)} серверов")
    logger.info(f"Все инструменты: {toolboxes}")
    return toolboxes

async def get_toolboxes():
    if not toolboxes:
        return await sync_toolboxes()
    return toolboxes

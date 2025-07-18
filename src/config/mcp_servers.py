from typing import List
import json
import smithery
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
import logging
from langchain_core.tools import Tool, StructuredTool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    
from src import schemas, enums
from src.config.settings import settings
from src.services.mcp_client_service import MCPClient

logger = logging.getLogger(__name__)

"""
Файл, в котором происходит синхронизация инструментов для тулбоксов.

MCP инструменты проходят через  mcp->langchain_mcp_adapters->langchain_core.tools
и по итогу схема тулза описана как json schema.

По этому, встроенные тулбоксы так же следует описывать в формате json schema.
"""


# словарь со списком удаленных mcp серверов
# id: схема тулбокса
mcp_servers = dict()

if settings.EVM_AGENT_KIT_SSE_URL:
    mcp_servers[enums.ToolboxList.EVM_KIT] = schemas.MCPSSEServer(
        name="EVM Agent Kit",
        description="Some blockchain tools",
        url=settings.EVM_AGENT_KIT_SSE_URL
    )

prebuild_toolboxes = []
if True:
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=6, source="news")
    news_search_api = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="json")          # уже BaseTool
    args_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string", 
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
    
    def search_tool(query: str):
        return news_search_api.run(query)
    
    search_tool = StructuredTool.from_function(
        func=search_tool,
        name="BASIC_SEARCH",
        description="Search news and basic snipets in internet. Feel free to use it to provide the user with up-to-date information. But do not use it if you have other search tools.",
        args_schema=args_schema
    )

    prebuilt_toolbox = schemas.Toolbox(
        id=enums.ToolboxList.BASIC,
        name="Basic tools",
        description="Tools for basic tasks.",
        tools=[search_tool],
        type=enums.ToolboxType.DEFAULT
    )
    prebuild_toolboxes.append(prebuilt_toolbox)

toolboxes: List[schemas.Toolbox] = []

async def sync_toolboxes():
    # Добавляем тулбоксы от mcp серверов
    for id, server in mcp_servers.items():
        mcp_client = MCPClient(server)
        async with mcp_client.get_session() as session:
            tools = await load_mcp_tools(session)
        toolbox = schemas.Toolbox(
            id=id,
            name=server.name,
            description=server.description,
            tools=tools,
            type=enums.ToolboxType.MCP
        )
        toolboxes.append(toolbox)
        
    # Добавляем тулбоксы от встроенных серверов
    toolboxes.extend(prebuild_toolboxes)
    
    logger.info(f"Синхронизировали инструменты для {len(mcp_servers)} серверов")
    logger.info(f"Все инструменты: {toolboxes}")
    return toolboxes

async def get_toolboxes():
    if not toolboxes:
        return await sync_toolboxes()
    return toolboxes

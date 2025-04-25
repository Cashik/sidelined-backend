import smithery


from src import schemas
from src.config import settings

mcp_servers = [
    schemas.MCPSSEServer(
        name="Thirdweb",
        description="Thirdweb tools provide access to the thirdweb platform. It's can be used to access onchain data, ask nebula etc.",
        url="http://thirdweb_mcp:8080/sse"
    )    
]

if settings.EXA_SEARCH_API_KEY and settings.SMITHERY_API_KEY:
    exa_search_url = smithery.create_smithery_url("wss://server.smithery.ai/exa/ws", {
        "exaApiKey": settings.EXA_SEARCH_API_KEY
    }) + f"&api_key={settings.SMITHERY_API_KEY}"
    mcp_servers.append(schemas.MCPWebSocketServer(
        name="Exa Search",
        description="Fast, intelligent web search and crawling.",
        url=exa_search_url
    ))
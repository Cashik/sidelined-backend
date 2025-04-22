import sys
import logging
from datetime import datetime
import os
import asyncio
from src.services.mcp_client_service import MCPClientService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from src.config import settings
from src.database import get_session
from src.models import *
from src.routers import auth, chat, user, requirements

app = FastAPI(
    title="2Eden API - Swagger UI",
    default_response_class=JSONResponse,
    prefix="/api"
)


@app.get("/health")
async def health_check():
    db = next(get_session())
    try:
        result = db.exec(text("SELECT 1 AS result")).first()
        db_status = "healthy" if result else "unhealthy"
    finally:
        db.close()

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": db_status,
            "version": settings.VERSION if hasattr(settings, 'VERSION') else "1.0.0"
        }
    )

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-New-Token"],
)

def get_db():
    db = next(get_session())
    try:
        yield db
    finally:
        db.close()


from mcp.types import ListToolsResult

# Функция для вывода доступных инструментов MCP сервера
async def list_mcp_tools():
    return {"status": "success", "message": "Тестовая фоновая задача запущена"}
    mcp_service = MCPClientService(
        name="MCP",
        description="MCP сервер",
        url="http://thirdweb_mcp:8080/sse"
    )
    await mcp_service.initialize()
    tools:ListToolsResult = await mcp_service.get_tools()
    
    
    logger.info(f"tool 1 Data: {tools[1].name} {tools[1].description} {tools[1].inputSchema} {tools[1].model_config}")
    
    #result = await mcp_service.invoke_tool(tools.tools[1].name, {"message": "Hello, world!"})
    #logger.info(f"result: {result}")


# Подключаем роутеры
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(user.router)
app.include_router(requirements.router)

@app.on_event("startup")
async def startup_event():
    # Выводим список инструментов MCP при старте сервера
    await list_mcp_tools()

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server at {settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, reload=True)

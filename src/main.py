import sys
import logging
from datetime import datetime
import os
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from fastapi.exceptions import RequestValidationError
from src.exceptions import BusinessError
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from src.config.settings import settings
from src.database import get_session
from src.models import *
from src.routers import auth, chat, user, subscription, yaps
from src.schemas import APIErrorContent, APIErrorResponse
from src.exceptions import APIError
from src.admin_starlette import setup_admin

app = FastAPI(
    title="Sidelined API - Swagger UI",
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

app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

# Подключаем роутеры
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(user.router)
app.include_router(subscription.router)
app.include_router(yaps.router)

# Инициализация SQLAdmin
setup_admin(app)

@app.on_event("startup")
async def startup_event():
    import src.config.mcp_servers
    await src.config.mcp_servers.sync_toolboxes()

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content=APIErrorResponse(
            error=APIErrorContent(
                code=exc.code,
                message=exc.message,
                details=exc.details
            )
        ).dict()
    )

@app.exception_handler(BusinessError)
async def business_error_handler(request: Request, exc: BusinessError):
    return JSONResponse(
        status_code=400,
        content=APIErrorResponse(
            error=APIErrorContent(
                code=exc.code,
                message=exc.message,
                details=exc.details
            )
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=APIErrorResponse(
            error=APIErrorContent(
                code="http_error",
                message=exc.detail,
                details=None
            )
        ).dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=APIErrorResponse(
            error=APIErrorContent(
                code="validation_error",
                message="Validation error",
                details=exc.errors()
            )
        ).dict()
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=APIErrorResponse(
            error=APIErrorContent(
                code="internal_error",
                message="Internal server error",
                details=None
            )
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server at {settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, reload=True)

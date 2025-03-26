import sys
import logging
from datetime import datetime

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
from src.routers import auth, chat, user

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
)

def get_db():
    db = next(get_session())
    try:
        yield db
    finally:
        db.close()

# Подключаем роутеры
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(user.router)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server at {settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, reload=True)

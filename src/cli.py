import typer
import asyncio
from sqlmodel import Session, select
from src.database import get_session
from src.models import User
from src import crud
from src.components.rank_system import RankSystem
from src.config import settings

app = typer.Typer()


if __name__ == "__main__":
    app()

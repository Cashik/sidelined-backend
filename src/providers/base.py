# providers/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from src.services.prompt_service import PromptService
from src import schemas 
from mcp import ClientSession

class AIProvider(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    async def generate_response(self, prompt_service: PromptService, tools: Optional[List[schemas.Tool]] = None) -> schemas.GeneratedResponse:
        raise NotImplementedError()
    
    @abstractmethod
    async def adapt_tools(self, tools: List[schemas.Tool]):
        raise NotImplementedError()

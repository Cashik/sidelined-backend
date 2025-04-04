
from enum import Enum


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Service(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    CLAUDE = "claude"
    

class Model(Enum):
    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    
    GEMINI_2_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro-exp-03-25"


class ChatStyle(Enum):
    FORMAL = "formal"
    INFORMAL = "informal"

class ChatDetailsLevel(Enum):
    HIGH = "detailed"
    MEDIUM = "to the point"
    LOW = "brief"

class ChainID(Enum):
    ETHEREUM = 1
    BASE = 8453

class TokenInterface(Enum):
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    
    
    


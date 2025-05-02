
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
    GPT_4O = "gpt-4o"
    GPT_4_1 = "gpt-4.1"
    GPT_O4_MINI = "o4-mini"
    
    GEMINI_2_5_PRO = "gemini-2.5-pro-preview-03-25"
    GEMINI_2_5_FLASH = "gemini-2.5-flash-preview-04-17"


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
    ARBITRUM = 42161

class TokenInterface(Enum):
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    
class MessageType(Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    LOCATION = "location"
    TRANSACTION = "transaction"
    


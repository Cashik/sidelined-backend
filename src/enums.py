
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


class ChatStyle(Enum):
    FORMAL = "formal"
    INFORMAL = "informal"

class ChatDetailsLevel(Enum):
    HIGH = "detailed"
    MEDIUM = "to the point"
    LOW = "brief"


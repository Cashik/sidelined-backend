from enum import Enum


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Service(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GEMINI = "Google Gemini"
    GROQ = "Groq"
    CLAUDE = "Claude"

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
    
class ToolboxType(Enum):
    DEFAULT = "default"
    MCP = "mcp"
    
class ToolboxList(Enum):
    BASIC = "basic"
    EVM_KIT = "evm_kit"

class SubscriptionPlanType(Enum):
    BASIC = "Basic"
    PRO = "Pro"
    ULTRA = "Ultra"

class PromoCodeType(Enum):
    PRO_UPGRADE = "pro_upgrade"

class ProjectAccountStatusType(Enum):
    MEDIA = "MEDIA"
    FOUNDER = "FOUNDER"
    
class SortType(Enum):
    NEW = "latest"
    POPULAR = "popular"

class AdminRole(Enum):
    ADMIN = "ADMIN"
    MODERATOR = "MODERATOR"


# ---------- AUTO-YAPS ---------- #
# personalization brain settings
class Tonality(str, Enum):
    NEUTRAL = "neutral"
    ENTHUSIASTIC = "enthusiastic"
    SKEPTICAL = "skeptical"
    IRONIC = "ironic"

class Formality(str, Enum):
    CASUAL = "casual"
    FORMAL = "formal"

class Perspective(str, Enum):
    FIRST_PERSON = "first_person"
    THIRD_PERSON = "third_person"

class LengthOption(str, Enum):
    SHORT = "short"       # ≤ 140 симв.
    NORMAL = "normal"     # ≤ 220 симв.
    LONG = "long"         # ≤ 280 симв.

class EmojiUsage(str, Enum):
    NO = "no"
    YES = "yes"

class HashtagPolicy(str, Enum):
    FORBIDDEN = "forbidden"
    REQUIRED = "required"

class MentionsPolicy(str, Enum):
    ANY = "any"
    DIRECT = "direct"
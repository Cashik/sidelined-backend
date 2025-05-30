from src import enums, schemas

all_ai_models = [
    schemas.AIModel(
        id=enums.Model.GPT_4O,
        provider=enums.Service.OPENAI,
        name="GPT-4o",
        description="Fast, intelligent, flexible GPT model"
    ),
    schemas.AIModel(
        id=enums.Model.GPT_4_1,
        provider=enums.Service.OPENAI,
        name="GPT-4.1",
        description="Flagship GPT model for complex tasks"
    ),
    schemas.AIModel(
        id=enums.Model.GPT_O4_MINI,
        provider=enums.Service.OPENAI,
        name="o4-mini",
        description="Fast reasoning model"
    ),
    schemas.AIModel(
        id=enums.Model.GEMINI_2_5_PRO,
        provider=enums.Service.GEMINI,
        name="Gemeni 2.5 Pro",
        description="Intelligent and flexible Gemini model"
    ),
    schemas.AIModel(
        id=enums.Model.GEMINI_2_5_FLASH,
        provider=enums.Service.GEMINI,
        name="Gemeni 2.5 Flash",
        description="Fast and flexible Gemini model"
    )
]
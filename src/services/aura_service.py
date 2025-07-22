from typing import List
import logging

from pydantic import BaseModel

from src.config import aura, settings
from src.enums import Model
from src.schemas import PostForAura


logger = logging.getLogger(__name__)


class AuraScoreResponse(BaseModel):
    posts_scores: List[aura.PostAuraScore]


def create_aura_system_prompt(posts: List[PostForAura]) -> str:
    """
    Creates a system prompt for the model to score posts based on Aura criteria.
    
    Args:
        posts: A list of posts to be scored.
        
    Returns:
        A string with the system prompt.
    """

    # Format criteria
    criteria_text = ""
    for c in aura.criteria_list:
        criteria_text += f"- **{c['id']} ({c['categoria']})**: {c['description']}\n"

    # Format posts
    posts_text = ""
    for post in posts:
        posts_text += f"POST ID: {post.id}\n"
        posts_text += f"POST TEXT: {post.text}\n\n"

    prompt = f"""
Your task is to evaluate the posts based on the given criteria.

RATING INSTRUCTIONS:
Rate each criterion on a scale of 0 to 10. Use the following guide to ensure consistency:
- **0: Completely Absent.** The post shows no evidence of this criterion.
- **1-2: Very Poor.** The criterion is attempted but executed extremely poorly, or is present in a way that is counter-productive.
- **3-4: Below Average.** A weak or superficial application of the criterion. It's recognizable but lacks any depth, skill, or impact.
- **5: Average.** A satisfactory but unremarkable application. The criterion is met in a basic, predictable way. Not good, not bad.
- **6-7: Above Average.** The criterion is applied skillfully and effectively. It adds clear value to the post but isn't a standout feature.
- **8-9: Excellent.** The criterion is a major strength of the post. It is executed with creativity, depth, and precision, making a significant impact.
- **10: Outstanding.** A perfect or near-perfect execution. The application of this criterion is masterful, defining the post's quality and making it exceptional.

CRITERIA FOR EVALUATION:
{criteria_text}

POSTS TO EVALUATE:
{posts_text}

OUTPUT FORMAT:
You must return a structured response in JSON format. The main object must contain a single key "posts_scores", which is a list.
Each item in the list is an object representing the score for a single post.
Each post score object must have:
1. "post_id": an integer ID of the post.
2. "scores": an object where keys are the criteria IDs (e.g., "insight", "reasoning") and values are integer scores (0-10).

The response must contain only JSON, without additional explanations.
    """
    return prompt


async def ask_ai_to_calculate_posts_aura_score(posts: List[PostForAura]) -> AuraScoreResponse:
    """
    Анализирует посты и запрашивает у Gemini оценку по критериям Aura.
    
    Args:
        posts: Список постов для оценки.
        
    Returns:
        AuraScoreResponse с результатами оценки.
    """
    if not posts:
        return AuraScoreResponse(posts_scores=[])
        
    from langchain.chat_models import init_chat_model
    from pydantic import ValidationError

    system_prompt = create_aura_system_prompt(posts)

    #logger.info(f"System prompt: {system_prompt}")

    # Определяем модель и провайдер (используем Gemini по умолчанию для анализа)
    model = Model.GEMINI_2_5_PRO
    model_provider = "google_genai"
    api_key = settings.get_settings().GEMINI_API_KEY
    
    # Инициализация LLM с structured output
    llm = init_chat_model(
        model=model.value,
        model_provider=model_provider,
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
    ).with_structured_output(AuraScoreResponse)
    
    try:
        result: AuraScoreResponse = await llm.ainvoke([
            {"role": "user", "content": system_prompt}
        ])
        return result
    except ValidationError as e:
        logger.error(f"Validation error in aura score analysis: {e}")
        # Повторная попытка с более строгим промптом
        strict_prompt = system_prompt + "\n\nIMPORTANT: Your response MUST be valid JSON that fully complies with the schema, otherwise an error will occur."
        result: AuraScoreResponse = await llm.ainvoke([
            {"role": "user", "content": strict_prompt}
        ])
        return result
    except Exception as e:
        logger.error(f"Failed to get response from AI model for aura score: {str(e)}")
        raise e 
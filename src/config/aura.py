from pydantic import BaseModel, Field, create_model, conint

# --- 1. Справочник критериев ---

criteria_list = []

criteria_list.append({
    "id": "insight",
    "categoria": "Original Thesis / Insight",
    "max_points": 20,
    "description": "Introduces a non-obvious idea, thesis, or reframe of a topic."
})

criteria_list.append({
    "id": "reasoning",
    "categoria": "Reasoning Depth",
    "max_points": 15,
    "description": "Includes multi-step logic, cause-effect thinking, comparisons, or inferences."
})

criteria_list.append({
    "id": "market",
    "categoria": "Market / Context Relevance",
    "max_points": 10,
    "description": "Anchored in current narratives, industry trends, or real-time cycles."
})

criteria_list.append({
    "id": "compression",
    "categoria": "Compression / Density",
    "max_points": 10,
    "description": "High signal-to-noise ratio; tweet compresses a lot of meaning into short form."
})

criteria_list.append({
    "id": "prediction",
    "categoria": "Prediction / Strategic Take",
    "max_points": 10,
    "description": "Makes a forward-looking call, strategic bet, or market forecast."
})

criteria_list.append({
    "id": "writing",
    "categoria": "Writing / Framing Power",
    "max_points": 10,
    "description": "Effective tone, structure, or phrasing. Hooky, readable, emotionally resonant."
})

criteria_list.append({
    "id": "discussion",
    "categoria": "Discussion Catalyst",
    "max_points": 10,
    "description": "Provokes quote tweets, thoughtful replies, or extended discourse."
})

criteria_list.append({
    "id": "structure",
    "categoria": "Conceptual Structure",
    "max_points": 10,
    "description": "Exhibits structure (e.g., “3 things”, layered argument, comparison, framework)."
})

criteria_list.append({
    "id": "voice",
    "categoria": "Cultural / Personal Voice",
    "max_points": 5,
    "description": "Reflects personality, experience, or resonates with a subculture."
})

"""
Создаем динамическую схему, которая включает все критерии из справочника,
чтобы не составлять эту схему вручную.

Данная схема нужна, чтобы ИИ не смог пропустить какой-то критерий при оценке.

т.е. при выставлении оценки ИИ обязан следовать схеме, в которой уже прописаны все критерии.
иначе он бы мог просто забыть про какой-то критерий, тк мы обрабатываем сразу несколько постов.
"""

Score = conint(ge=0, le=10)

fields = {
    c["id"]: (Score, Field(..., description=c["description"]))
    for c in criteria_list
}

Criterias = create_model("Criterias", **fields) # type: ignore[valid-type]

class PostAuraScore(BaseModel):
    post_id: int
    scores:  Criterias
import logging
from typing import List

from src import schemas
from src.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

NEBULA_FUNCTIONS_TEMPLATE = f"""
2. Interaction of the user with the blockchain.  
Nebula is an AI agent with access to blockchain data. You can interact with it to supplement the answer to the user.

2.1 Data retrieval.  
If the user asks a question related to obtaining blockchain data, simply forward this message to your colleague. Keep in mind that Nebula does not understand the context of your conversation with the user, so if necessary, you must modify the user's request.

call nebula_ask(str)

2.2 Creating a transaction.  
Nebula can create a transaction for the user, which they will only need to sign. As with data retrieval, you need to ensure that Nebula has all the necessary context.

call nebula_sign(str)

Usage examples:  
call nebula_ask("native or simplified user request")
call nebula_sign("native or supplemented user request")
"""


# todo: add nebula tools
nebula_tools = []

class PromptService:
    def __init__(self, generate_data: schemas.AssistantGenerateData):
        self.generate_data = generate_data

    def generate_system_prompt(self) -> str:
        # Системное сообщение (инструкции для модели)
        system_message = f"You are a helpful assistant."
        
        if self.generate_data.user.preferred_name or self.generate_data.user.user_context or self.generate_data.chat_settings.chat_style or self.generate_data.chat_settings.chat_details_level:
            system_message += "\nUser preferences:"
            if self.generate_data.user.preferred_name:
                system_message += f"""\n- want to be called as "{self.generate_data.user.preferred_name}"."""
            if self.generate_data.chat_settings.chat_style:
                system_message += f"\n- wants to communicate in the following style: {self.generate_data.chat_settings.chat_style.value}."
            if self.generate_data.chat_settings.chat_details_level:
                system_message += f"\n- wants your answers to be as detailed as this: {self.generate_data.chat_settings.chat_details_level.value}."
            if self.generate_data.user.user_context:
                system_message += f"\nUser added some information about himself: \n{self.generate_data.user.user_context}."

            system_message += "\n\n"
            
        if settings.FUNCTIONALITY_ENABLED:
            if settings.FACTS_FUNCTIONALITY_ENABLED:
                system_message += "\nFrom previous or current conversation you have some notes about the user. Remember that the user's information is constantly evolving."
                system_message += "\nIMPORTANT: At every new interaction, ALWAYS verify and update (or delete, if necessary) all facts about the user using your available functions!"
                system_message += "\nInformation to focus on: any details provided by the user — including but not limited to locations, tasks, preferences, list of close people and their activities, hobbies, work, life goals, events, and any other information."
                system_message += "\nCurrent list of notes (id - note):"
                if self.generate_data.user.facts:
                    for fact in self.generate_data.user.facts:
                        system_message += f"\n{fact.id} - {fact.description}"
                else:
                    system_message += "\nList is empty."
                system_message += "\nRules:"
                system_message += "\n1. Record any new information about the user, including seemingly minor details."
                system_message += "\n2. Base new notes on the user's messages or on your verified findings. Cross-check with existing data to update outdated facts."
                system_message += "\n3. The user is aware of this list, so do not mention it in your answers."
                system_message += "\n4. Immediately delete or update any outdated or irrelevant notes."
                system_message += "\n5. Always check for changes: if new information alters previous notes, update them without delay."
                system_message += "\n6. When adding new facts, always keep them concise and summarized, avoiding excessive details."

            if settings.NEBULA_FUNCTIONALITY_ENABLED:
                system_message += f"\n{NEBULA_FUNCTIONS_TEMPLATE}"
        
        logger.info(f"System message: {system_message}")
        return system_message
    
    def get_available_tools(self) -> List[dict]:
        available_tools = []
        if settings.FACTS_FUNCTIONALITY_ENABLED:
            available_tools.extend(facts_tools)
        if settings.NEBULA_FUNCTIONALITY_ENABLED:
            available_tools.extend(nebula_tools)
        return available_tools


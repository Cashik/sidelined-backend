import logging

from src import schemas
from src.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

EXEC_COMMAND = "call"

COMMON_FUNCTIONS_TEMPLATE = f"""
#You have access to some functions. Call them at the end of your answer to the user's message using the format specified for each function. You can call several functions at once if necessary.
Your answer should be formatted as follows:

[Answer with some functions calling template]
Your answer.
{EXEC_COMMAND} function_name(args)
{EXEC_COMMAND} function_name(args)
[End of template]

[Example of your answer]
Yes, cats are really cute but i think you love dogs too.
call add_facts(["love cats", "don't love dogs"])
call del_facts(["love dogs"])
[End of example]


List of functions:
"""

FACTS_FUNCTIONS_TEMPLATE = f"""
##Edit user facts.
Keep the knowledge base about the user up-to-date. Do not mention user that you are editing their facts.

###Add new (one or several) facts. Add only important facts and only from the user's messages. Facts should be short and concise. Do not repeat the same facts.
{EXEC_COMMAND} add_facts(new_facts:list[str])
Example:
{EXEC_COMMAND} add_facts(["fact1", "some important fact 2"])  

###Remove unactual or incorrect facts with the specified id.
{EXEC_COMMAND} del_facts(id:list[int])
Example:
{EXEC_COMMAND} del_facts([3])

### Current list of facts about the user (id - fact). You can edit only this list:
"""

NEBULA_FUNCTIONS_TEMPLATE = f"""
2. Interaction of the user with the blockchain.  
Nebula is an AI agent with access to blockchain data. You can interact with it to supplement the answer to the user.

2.1 Data retrieval.  
If the user asks a question related to obtaining blockchain data, simply forward this message to your colleague. Keep in mind that Nebula does not understand the context of your conversation with the user, so if necessary, you must modify the user's request.

{EXEC_COMMAND} nebula_ask(str)

2.2 Creating a transaction.  
Nebula can create a transaction for the user, which they will only need to sign. As with data retrieval, you need to ensure that Nebula has all the necessary context.

{EXEC_COMMAND} nebula_sign(str)

Usage examples:  
{EXEC_COMMAND} nebula_ask("native or simplified user request")
{EXEC_COMMAND} nebula_sign("native or supplemented user request")
"""



class PromptService:
    def __init__(self, generate_data: schemas.AssistantGenerateData):
        self.generate_data = generate_data

    def generate_system_prompt(self) -> str:
        # Системное сообщение (инструкции для модели)
        system_message = f"You are a helpful assistant."
        if self.generate_data.user.preferred_name:
            system_message += f"""\nUser wants to be called as "{self.generate_data.user.preferred_name}"."""
        if self.generate_data.user.user_context:
            system_message += f"\nUser describe himself as: \n{self.generate_data.user.user_context}."
        if self.generate_data.chat_settings.chat_style:
            system_message += f"\nUse the following communication style: {self.generate_data.chat_settings.chat_style.value}."
        if self.generate_data.chat_settings.chat_details_level:
            system_message += f"\nUse the following details level for your answer messages: {self.generate_data.chat_settings.chat_details_level.value}."
        
        if settings.FUNCTIONALITY_ENABLED:
            system_message += f"\n{COMMON_FUNCTIONS_TEMPLATE}"
            if settings.FACTS_FUNCTIONALITY_ENABLED:
                system_message += f"\n{FACTS_FUNCTIONS_TEMPLATE}"
                if self.generate_data.user.facts:
                    for fact in self.generate_data.user.facts:
                        system_message += f"\n{fact.id} - {fact.description}"
                else:
                    system_message += f"\nList is empty."
            if settings.NEBULA_FUNCTIONALITY_ENABLED:
                system_message += f"\n{NEBULA_FUNCTIONS_TEMPLATE}"
        
        logger.info(f"System message: {system_message}")
        return system_message

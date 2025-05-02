import logging
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

from src import schemas, enums
from src.config import settings

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


class PromptService:
    def __init__(self, generate_data: schemas.AssistantGenerateData):
        self.generate_data: schemas.AssistantGenerateData = generate_data

    def generate_system_prompt(self) -> str:
        # Системное сообщение (инструкции для модели)
        system_message = f"You are a helpful assistant."
        
        system_message += f"\n#System information about user:"
        if self.generate_data.user.preferred_name or self.generate_data.user.user_context or self.generate_data.chat_settings.chat_style or self.generate_data.chat_settings.chat_details_level:
            system_message += "\n##User preferences and account information:"
            if self.generate_data.user.preferred_name:
                system_message += f"""\n- want to be called as "{self.generate_data.user.preferred_name}"."""
            if self.generate_data.chat_settings.chat_style:
                system_message += f"\n- preferred chat style: {self.generate_data.chat_settings.chat_style.value}."
            if self.generate_data.chat_settings.chat_details_level:
                system_message += f"\n- desired level of detail in your answers: {self.generate_data.chat_settings.chat_details_level.value}."
            if self.generate_data.user.user_context:
                system_message += f"\nUser added some information about himself: \n{self.generate_data.user.user_context}."

            system_message += "\n"
        
        if self.generate_data.user.addresses:
            system_message += f"\n##User web3 information:"
            if self.generate_data.user.selected_address:
                system_message += f"\nUser current address: {self.generate_data.user.selected_address}. Use it for if user do not specify another address."
            other_addresses = [address for address in self.generate_data.user.addresses if address != self.generate_data.user.selected_address]
            if other_addresses:
                system_message += f"\nAll user connected addresses:" + ", ".join([f'"{address}"' for address in other_addresses])
        
        system_message += "\nFor all tools you must use Base network(chain_id: 8453)."
        system_message += "\nFor response use language that user uses in his messages.\n"
        
        if settings.FUNCTIONALITY_ENABLED:
            if settings.FACTS_FUNCTIONALITY_ENABLED:
                system_message += "\nFrom previous conversation you have some notes about the user."
                system_message += "\nJust use it to answer user's questions more precisely."
                system_message += "\nDO NOT TRY TO CHANGE THIS LIST."
                system_message += "\nCurrent list of notes:"
                if self.generate_data.user.facts:
                    for fact in self.generate_data.user.facts:
                        system_message += f"\n- {fact.description}"
                else:
                    system_message += "\nList is empty."

        logger.info(f"System message: {system_message}")
        return system_message
    
    def generate_langchain_messages(self, avoid_system_role: bool = False) -> list[BaseMessage]:
        """
        If avoid_system_role is True, system role will be added as human message. 
        This is useful for some old models that do not support system role.
        """
        messages = []
        system_prompt = self.generate_system_prompt()
        if avoid_system_role:
            messages.append(HumanMessage(content="System: " + system_prompt))
        else:
            messages.append(SystemMessage(content=system_prompt))
            
        
        # Добавление сообщений из истории чата
        for nonce in sorted(self.generate_data.chat.messages.keys()):
            message = max(self.generate_data.chat.messages[nonce], key=lambda x: x.selected_at)
            
            if message.type is not enums.MessageType.TEXT:
                continue
            
            if message.sender is enums.Role.USER:
                messages.append(HumanMessage(content=message.content.message))
            elif message.sender is enums.Role.ASSISTANT:
                messages.append(AIMessage(content=message.content.message))
            elif message.sender is enums.Role.SYSTEM:
                if avoid_system_role:
                    messages.append(HumanMessage(content="System: " + message.content.message))
                else:
                    messages.append(SystemMessage(content=message.content.message))
            else:
                logger.error(f"Unknown message sender: {message.sender}")
                pass
        return messages

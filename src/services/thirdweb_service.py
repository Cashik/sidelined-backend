from pydantic import BaseModel
import requests
import logging


from src import enums, schemas, exceptions
from src.config import settings

logger = logging.getLogger(__name__)

class TokenBalance(BaseModel):
    chain_id: int
    address_lower: str
    balance: int

class NebulaContext(BaseModel):
    chain_ids: list[int] | None = None
    addresses: list[str] | None = None

class NebulaChatRequest(BaseModel):
    message: str
    session_id: str|None=None
    context: NebulaContext|None=None

class NebulaChatResponse(BaseModel):
    session_id: str
    request_id: str
    message: str


class ThirdwebService():
    
    general_api_host = "https://insight.thirdweb.com"
    nebula_api_host = "https://nebula-api.thirdweb.com"
    allowed_interfaces = ["erc20", "erc721", "erc1155"]
    query_limit = 50
    DEFAULT_USER_ID = "default-user"
    
    def __init__(self, app_id: str, private_key: str):
        self.app_id = app_id
        self.private_key = private_key
    
    async def get_balances(self, owner_address: str, chain_id: int, interface: enums.TokenInterface) -> list[TokenBalance]:
        if interface != enums.TokenInterface.ERC20:
            raise NotImplementedError(f"Currently only ERC20 tokens are supported. Found {interface.value} interface.")
        
        if interface.value not in self.allowed_interfaces:
            raise NotImplementedError(f"Currently only {self.allowed_interfaces} interfaces are supported. Found {interface.value} interface.")
        
        url = f"{self.general_api_host}/v1/tokens/{interface.value}/{owner_address}"
        
        query_params = {
            "chain": chain_id,
            "limit": self.query_limit,
            "metadata": "false",
            "include_spam": "true",
        }
        headers = {
            'x-client-id': self.app_id
        }   
        try:
            response = requests.get(url, params=query_params, headers=headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            # TODO: 
            logger.error(f"Error getting balances with params: {query_params} and error: {e}")
            raise exceptions.ThirdwebServiceException()
        
        request_balances = data["data"]
        result_balances = []
        for balance in request_balances:
            result_balances.append(
                TokenBalance(
                    chain_id=balance["chain_id"],
                    address_lower=balance["token_address"].lower(),
                    balance=balance["balance"]
                )
            )
        
        return result_balances
    
    async def nebula_chat(self, request: NebulaChatRequest) -> NebulaChatResponse:
        url = f"{self.nebula_api_host}/chat"
        headers = {
            'x-client-id': self.app_id,
            'x-secret-key': self.private_key
        }
        data = {
            "message": request.message,
            "stream": False,
            "user_id": self.DEFAULT_USER_ID
        }
        if request.session_id:
            data["session_id"] = request.session_id
        if request.context:
            data["context"] = request.context.model_dump()
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error getting nebula response {e}")
            raise exceptions.ThirdwebServiceException()
        
        response = NebulaChatResponse(
            session_id=data["session_id"],
            request_id=data["request_id"],
            message=data["message"]
        )
        return response


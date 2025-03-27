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
    

class ThirdwebService():
    
    host = "https://insight.thirdweb.com"
    allowed_interfaces = ["erc20", "erc721", "erc1155"]
    query_limit = 50
    
    def __init__(self, app_id: str, private_key: str):
        self.app_id = app_id
        self.private_key = private_key
    
    async def get_balances(self, owner_address: str, chain_id: int, interface: enums.TokenInterface) -> list[TokenBalance]:
        if interface != enums.TokenInterface.ERC20:
            raise NotImplementedError(f"Currently only ERC20 tokens are supported. Found {interface.value} interface.")
        
        if interface.value not in self.allowed_interfaces:
            raise NotImplementedError(f"Currently only {self.allowed_interfaces} interfaces are supported. Found {interface.value} interface.")
        
        url = f"{self.host}/v1/tokens/{interface.value}/{owner_address}"
        
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

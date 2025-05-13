from src import enums
from web3 import Web3

erc20_abi = [
    {
        "name": "balanceOf",
        "type": "function",
        "constant": True,
        "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}],
        "outputs": [{"name": "balance", "type": "uint256"}],
    }
]

erc721_abi = [
    {
        "name": "balanceOf",
        "type": "function",
        "constant": True,
        "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}],
        "outputs": [{"name": "balance", "type": "uint256"}],
    },
]

class Web3Service():
    
    BASE_URL = "rpc.ankr.com"
    
    NETWORK_TO_URL = {
        enums.ChainID.BASE: "base",
        enums.ChainID.ARBITRUM: "arbitrum",
        enums.ChainID.ETHEREUM: "eth",
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        

    def _get_rpc_http_url(self, chain_id: enums.ChainID):
        return f"https://{self.BASE_URL}/{self.NETWORK_TO_URL[chain_id]}/{self.api_key}"
    
    def _get_rpc_ws_url(self, chain_id: enums.ChainID):
        return f"wss://{self.BASE_URL}/{self.NETWORK_TO_URL[chain_id]}/{self.api_key}"
    
    def get_ERC20_balance(self, token_address: str, user_address: str, chain_id: enums.ChainID):
        token_address = Web3.to_checksum_address(token_address)
        user_address = Web3.to_checksum_address(user_address)
        self.web3 = Web3(Web3.HTTPProvider(self._get_rpc_http_url(chain_id)))
        erc20 = self.web3.eth.contract(address=token_address, abi=erc20_abi)
        raw_balance   = erc20.functions.balanceOf(user_address).call()
        return raw_balance
    
    def get_ERC721_balance(self, token_address: str, user_address: str, chain_id: enums.ChainID):
        token_address = Web3.to_checksum_address(token_address)
        user_address = Web3.to_checksum_address(user_address)
        self.web3 = Web3(Web3.HTTPProvider(self._get_rpc_http_url(chain_id)))
        erc721 = self.web3.eth.contract(address=token_address, abi=erc721_abi)
        raw_balance   = erc721.functions.balanceOf(user_address).call()
        return raw_balance
    
    def raw_balance_to_human(self, raw_balance: int, decimals: int) -> float:
        return raw_balance / (10**decimals)
    
    


from typing import Dict, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field
from datetime import datetime
import time

from src import enums


class LoginPayloadRequest(BaseModel):
    address: str
    chainId: int

class LoginPayload(BaseModel):
    domain: str
    address: str
    statement: str
    uri: str
    version: str
    chain_id: int
    nonce: str
    issued_at: str
    expiration_time: str
    
class LoginRequest(BaseModel):
    payload: LoginPayload
    signature: str
    
class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    
class IsLoginResponse(BaseModel):
    logged_in: bool


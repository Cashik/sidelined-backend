from eth_account.messages import encode_defunct
from eth_account import Account
from eth_utils import to_checksum_address, is_address, is_checksum_address
import json
import logging
from datetime import datetime, timedelta, timezone
import random
import base64
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError
import base58

from src import schemas, enums

logger = logging.getLogger(__name__)

SOLANA_NETWORKS_IDS = [101]



def verify_signature(payload: schemas.LoginPayload, signature: str) -> bool:
    """
    Проверяет, что подпись была создана владельцем указанного адреса
    с использованием переписанного кода из клиенской библиотеки thirdweb 
    
    Args:
        payload: данные, которые были подписаны
        signature: подпись в формате hex
        
    Returns:
        True если подпись валидна, False в противном случае
    """
    
    if payload.chain_id in SOLANA_NETWORKS_IDS:
        family = enums.ChainFamily.SOLANA
    else:
        family = enums.ChainFamily.EVM
    
    try:
        logger.info(f"Verifying signature for address: {payload.address}")
        logger.info(f"Payload data: {payload.dict()}")
        
        # Точное воспроизведение логики из кода thirdweb
        # 1. Формируем заголовок и адрес
        if family == enums.ChainFamily.EVM:
            type_field = "Ethereum"
        elif family == enums.ChainFamily.SOLANA:
            type_field = "Solana"
        else:
            raise ValueError(f"Invalid chain family: {family}")
        
        header = f"{payload.domain} wants you to sign in with your {type_field} account:"
        # prefix = [header, payload.address].join("\n") - JavaScript
        prefix = "\n".join([header, payload.address])
        
        # 2. Добавляем statement с двойным переносом строки
        # prefix = [prefix, payload.statement].join("\n\n") - JavaScript
        prefix = f"{prefix}\n\n{payload.statement}"
        
        # 3. Добавляем перенос строки после statement, если он есть
        if payload.statement:
            prefix += "\n"
            
        # 4. Создаем массив для суффикса
        suffix_array = []
        
        # 5. Добавляем URI если оно есть
        if payload.uri:
            suffix_array.append(f"URI: {payload.uri}")
            
        # 6. Добавляем версию
        suffix_array.append(f"Version: {payload.version}")
        
        # 7. Добавляем Chain ID если оно есть, иначе "1"
        if payload.chain_id:
            suffix_array.append(f"Chain ID: {payload.chain_id}")
        
        # 8. Добавляем nonce, issued_at, expiration_time
        suffix_array.append(f"Nonce: {payload.nonce}")
        suffix_array.append(f"Issued At: {payload.issued_at}")
        suffix_array.append(f"Expiration Time: {payload.expiration_time}")
        
        # 9. Формируем суффикс, объединяя элементы с переносом строки
        suffix = "\n".join(suffix_array)
        
        # 10. Объединяем prefix и suffix с ОДНИМ переносом строки между ними
        # JavaScript: return [prefix, suffix].join("\n")
        message_to_sign = f"{prefix}\n{suffix}"
        
        logger.info(f"Generated message exactly matching thirdweb format: {message_to_sign}")
        
        
        if family == enums.ChainFamily.EVM:
            # Создаем сообщение для проверки
            message = encode_defunct(text=message_to_sign)
            
            # Получаем адрес из подписи
            recovered_address = Account.recover_message(message, signature=signature)
            logger.info(f"Recovered address: {recovered_address}")
            
            # Сравниваем восстановленный адрес с адресом в payload
            is_valid = to_checksum_address(recovered_address) == to_checksum_address(payload.address)
            
            if is_valid:
                logger.info("Signature verification successful")
            else:
                logger.warning(f"Signature verification failed for address {payload.address}")
                logger.warning(f"Expected: {payload.address}, got: {recovered_address}")
        elif family == enums.ChainFamily.SOLANA:
            try:
                logger.info("Verifying Solana signature")
                
                # Декодируем адрес (публичный ключ) из base58
                public_key_bytes = base58.b58decode(payload.address)
                
                # Создаем объект VerifyKey
                verify_key = VerifyKey(public_key_bytes)
                
                # Декодируем подпись из base64, как это делает фронтенд
                signature_bytes = base64.b64decode(signature)
                
                # Кодируем сообщение в байты
                message_bytes = message_to_sign.encode('utf-8')
                
                # Проверяем подпись. Если подпись неверна, будет выброшено исключение.
                verify_key.verify(message_bytes, signature_bytes)
                
                logger.info(f"Solana signature verification successful for address {payload.address}")
                is_valid = True
            except BadSignatureError:
                logger.warning(f"Invalid Solana signature for address {payload.address}")
                is_valid = False
            except Exception as e:
                logger.error(f"An unexpected error occurred during Solana signature verification: {e}")
                is_valid = False
                
        return is_valid
        
    except Exception as e:
        logger.error(f"Error verifying signature: {str(e)}")
        return False


def is_valid_eth_address(address: str) -> bool:
    """
    Проверяет, что строка является валидным Ethereum адресом
    
    Args:
        address: Ethereum адрес для проверки
        
    Returns:
        True если адрес валиден, False в противном случае
    """
    try:
        # Проверка формата адреса
        if not is_address(address):
            logger.warning(f"Invalid Ethereum address format: {address}")
            return False
        
        # Дополнительная проверка для EIP-55 совместимости
        if not is_checksum_address(address):
            logger.warning(f"Address {address} is not in checksum format")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating address: {str(e)}")
        return False

def is_valid_solana_address(address: str) -> bool:
    """
    Проверяет, что строка является валидным Solana адресом (формат Base58)
    """
    try:
        # TODO: использовать библиотеку solana-py или тд для проверки адреса
        
        # Проверяем, что адрес не в lowercase
        if address.lower() == address:
            logger.warning(f"Solana address is in lowercase: {address}")
            return False
        
        # Попытка декодирования из Base58
        decoded_bytes = base58.b58decode(address)
        # Публичный ключ в Solana имеет длину 32 байта
        if len(decoded_bytes) == 32:
            return True
        logger.warning(f"Invalid Solana address length: {len(decoded_bytes)} bytes for address {address}")
        return False
    except Exception:
        logger.warning(f"Invalid Base58 format for Solana address: {address}")
        return False


def create_login_payload(
    address: str,
    chain_id: int,
    domain: str = "sidelined.ai",
    statement: str = "Please sign this message to login to Sidelined AI",
    uri: str = "https://sidelined.ai",
    version: str = "1",
    expiration_hours: int = 24
) -> schemas.LoginPayload:
    """
    Создает payload для подписи при логине или добавлении адреса
    
    Args:
        address: Ethereum адрес
        chain_id: ID сети
        domain: Домен приложения
        statement: Сообщение для подписи
        uri: URI приложения
        version: Версия payload
        expiration_hours: Время жизни payload в часах
        
    Returns:
        LoginPayload: Созданный payload
    """
    issued_at = datetime.utcnow().isoformat()
    expiration_time = (datetime.utcnow() + timedelta(hours=expiration_hours)).isoformat()
    
    return schemas.LoginPayload(
        domain=domain,
        address=address,
        statement=statement,
        uri=uri,
        version=version,
        chain_id=chain_id,
        nonce=str(random.randint(0, 1000000)),
        issued_at=issued_at,
        expiration_time=expiration_time
    )

def validate_payload(payload: schemas.LoginPayload) -> bool:
    """
    Проверяет валидность payload'а
    
    Args:
        payload: Payload для проверки
        
    Returns:
        bool: True если payload валиден, False в противном случае
    """
    try:
        if payload.chain_id in SOLANA_NETWORKS_IDS:
            family = enums.ChainFamily.SOLANA
        else:
            family = enums.ChainFamily.EVM

        # Проверяем валидность адреса в зависимости от семейства
        if family == enums.ChainFamily.EVM:
            if not is_valid_eth_address(payload.address):
                return False
        elif family == enums.ChainFamily.SOLANA:
            if not is_valid_solana_address(payload.address):
                return False
        else:
            # На случай, если появятся новые семейства, а логика не будет добавлена
            logger.error(f"Unknown chain family for address validation: {family}")
            return False
            
        # Проверяем время
        issued_at = datetime.fromisoformat(payload.issued_at.replace('Z', '+00:00'))
        expiration_time = datetime.fromisoformat(payload.expiration_time.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        
        if now < issued_at or now > expiration_time:
            logger.warning(f"Payload time validation failed: now={now}, issued_at={issued_at}, expiration_time={expiration_time}")
            return False
            
        # Проверяем обязательные поля
        if not all([
            payload.domain,
            payload.statement,
            payload.uri,
            payload.version,
            payload.nonce,
            payload.chain_id
        ]):
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating payload: {str(e)}")
        return False



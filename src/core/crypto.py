from eth_account.messages import encode_defunct
from eth_account import Account
from eth_utils import to_checksum_address, is_address, is_checksum_address
import json
import logging
# Импортируем библиотеку SIWE
from siwe import SiweMessage

from src import schemas

logger = logging.getLogger(__name__)

def verify_signature(payload: schemas.LoginPayload, signature: str) -> bool:
    """
    Проверяет, что подпись была создана владельцем указанного адреса
    с использованием стандарта SIWE (Sign-In with Ethereum)
    
    Args:
        payload: данные, которые были подписаны
        signature: подпись в формате hex
        
    Returns:
        True если подпись валидна, False в противном случае
    """
    try:
        logger.info(f"Verifying signature for address: {payload.address}")
        logger.info(f"Payload data: {payload.dict()}")
        
        # Точное воспроизведение логики из кода thirdweb
        # 1. Формируем заголовок и адрес
        type_field = "Ethereum"
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
            
        return is_valid
        
    except Exception as e:
        logger.error(f"Error verifying signature: {str(e)}")
        return False


def is_valid_address(address: str) -> bool:
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
            
        # Преобразуем в checksum формат и проверяем
        checksum_address = to_checksum_address(address)
        
        # Дополнительная проверка для EIP-55 совместимости
        if not is_checksum_address(checksum_address):
            logger.warning(f"Address {address} is not in checksum format")
            # Мы все равно возвращаем True, так как формат адреса корректный,
            # но это может быть полезно для логирования
        
        return True
    except Exception as e:
        logger.error(f"Error validating address: {str(e)}")
        return False



from fastapi import HTTPException, status


class ChatNotFoundException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Чат не найден"
        )


class UserNotChatOwnerException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="У вас нет доступа к этому чату"
        )


class InvalidNonceException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нельзя изменить сообщение ассистента"
        )

class ThirdwebServiceException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при получении балансов"
        )

class FactNotFoundException(Exception):
    """Исключение, возникающее когда факт не найден"""
    pass

class MessageNotFoundException(Exception):
    pass

class InvalidMessageTypeException(Exception):
    pass

class AddressAlreadyExistsException(Exception):
    """Исключение, возникающее при попытке добавить уже существующий адрес"""
    pass

class AddressNotFoundException(Exception):
    """Исключение, возникающее при попытке удалить несуществующий адрес"""
    pass

class LastAddressException(Exception):
    """Исключение, возникающее при попытке удалить последний адрес пользователя"""
    pass


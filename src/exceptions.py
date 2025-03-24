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

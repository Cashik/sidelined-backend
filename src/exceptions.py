class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details

class BusinessError(Exception):
    def __init__(self, code: str, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details

class ChatNotFoundError(BusinessError):
    def __init__(self):
        super().__init__(code="chat_not_found", message="Chat not found")

class UserNotChatOwnerError(BusinessError):
    def __init__(self):
        super().__init__(code="not_chat_owner", message="You don't have access to this chat")

class InvalidNonceError(BusinessError):
    def __init__(self):
        super().__init__(code="invalid_nonce", message="You can't change assistant message")

class ThirdwebServiceError(BusinessError):
    def __init__(self):
        super().__init__(code="thirdweb_service_error", message="Error while getting balances")

class Web3ServiceError(BusinessError):
    def __init__(self):
        super().__init__(code="web3_service_error", message="Error while getting balances")

class FactNotFoundError(BusinessError):
    def __init__(self):
        super().__init__(code="fact_not_found", message="Fact not found")

class MessageNotFoundError(BusinessError):
    def __init__(self):
        super().__init__(code="message_not_found", message="Message not found")

class InvalidMessageTypeError(BusinessError):
    def __init__(self):
        super().__init__(code="invalid_message_type", message="Invalid message type")

class AddressAlreadyExistsError(BusinessError):
    def __init__(self):
        super().__init__(code="address_already_exists", message="Address already exists")

class AddressNotFoundError(BusinessError):
    def __init__(self):
        super().__init__(code="address_not_found", message="Address not found")

class LastAddressError(BusinessError):
    def __init__(self):
        super().__init__(code="last_address", message="You can't delete the last address")

class AddMessageError(BusinessError):
    def __init__(self):
        super().__init__(code="add_message_error", message="Error while adding message to chat")


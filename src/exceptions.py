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

class UserNotFoundError(BusinessError):
    def __init__(self):
        super().__init__(code="user_not_found", message="User not found")

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
    def __init__(self, message: str = "Fact not found", code: str = "fact_not_found"):
        super().__init__(message=message, code=code)

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

class PromoCodeActivationError(BusinessError):
    def __init__(self, message: str, code: str = "promo_code_activation_failed"):
        super().__init__(message=message, code=code)

class PromoCodeNotFoundError(BusinessError):
    def __init__(self):
        super().__init__(code="promo_code_not_found", message="Promo code not found")

class AutoYapsError(BusinessError):
    def __init__(self, message: str):
        super().__init__(code="auto_yaps_error", message=message)
        
class XAccountNotLinkedError(BusinessError):
    def __init__(self):
        super().__init__(code="x_account_not_linked", message="Your X account is not linked to your account.")

class PostTextExtractionError(BusinessError):
    def __init__(self, message: str = "Could not extract text from post JSON", code: str = "post_text_extraction_error"):
        super().__init__(message=message, code=code)


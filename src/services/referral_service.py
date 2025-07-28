from typing import Optional
from hashids import Hashids
from src.config.settings import settings

class ReferralService:
    def __init__(self):
        self.hashids = Hashids(salt=settings.SECRET_KEY, min_length=8)

    def generate_referral_code(self, user_id: int) -> str:
        """Генерирует реферальный код для пользователя."""
        return self.hashids.encode(user_id)

    def decode_referral_code(self, code: str) -> Optional[int]:
        """Декодирует реферальный код и возвращает ID пользователя."""
        decoded = self.hashids.decode(code)
        if decoded:
            return decoded[0]
        return None

referral_service = ReferralService() 
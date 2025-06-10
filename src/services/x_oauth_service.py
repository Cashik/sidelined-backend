import httpx
from urllib.parse import urlencode
from src.config.settings import settings

class XOAuthService:
    AUTH_URL = "https://twitter.com/i/oauth2/authorize"
    TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
    USERINFO_URL = "https://api.twitter.com/2/users/me"
    
    def __init__(self):
        if not settings.TWITTER_CLIENT_ID or not settings.TWITTER_CLIENT_SECRET:
            raise ValueError("Twitter OAuth credentials are not set")
        self.client_id = settings.TWITTER_CLIENT_ID
        self.client_secret = settings.TWITTER_CLIENT_SECRET
        self.redirect_uri = f"{settings.TWITTER_REDIRECT_URI}/yaps/x/callback"  # Указать свой redirect_uri
        self.scope = "tweet.read users.read offline.access"  # Минимально необходимый scope

    def get_authorize_url(self, state: str) -> str:
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scope,
            "state": state,
            "code_challenge": "challenge",  # Для PKCE, если нужно
            "code_challenge_method": "plain",  # Для PKCE, если нужно
        }
        return f"{self.AUTH_URL}?{urlencode(params)}"

    async def fetch_token(self, code: str) -> dict:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": "challenge",  # Для PKCE, если нужно
        }
        auth = (self.client_id, self.client_secret)
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.TOKEN_URL, data=data, auth=auth)
            resp.raise_for_status()
            return resp.json()

    async def get_user_login(self, access_token: str) -> str:
        headers = {"Authorization": f"Bearer {access_token}"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(self.USERINFO_URL, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["data"]["username"] 
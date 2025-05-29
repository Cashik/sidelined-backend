import os
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid
import aiohttp
from pydantic import BaseModel, Field
import logging
import json

from src.config.settings import settings
from src.utils_base import parse_date_to_timestamp, timestamp_to_X_date

logger = logging.getLogger(__name__)


class UserLegacy(BaseModel):
    name:          str
    screen_name:   str
    followers_count: int
    profile_image_url_https: str



class User(BaseModel):
    """`core › user_results › result`"""
    rest_id: str
    legacy: UserLegacy


class TweetLegacy(BaseModel):
    id_str: str
    full_text: str
    created_at: str
    favorite_count: int
    retweet_count: int
    reply_count: int
    lang: str



class TweetViews(BaseModel):
    count: Optional[str] = None
    state: str


class UserResults(BaseModel):
    """`core › user_results` wrapper"""
    result: User


class TweetCore(BaseModel):
    user_results: UserResults


class TweetResult(BaseModel):
    """`tweet_results › result`"""
    rest_id: str
    core: TweetCore
    legacy: TweetLegacy
    views: Optional[TweetViews] = None


class UnknownTweetResult(BaseModel):
    """Заглушка для твитов без обязательных полей (rest_id / core / legacy)."""

    limitedActionResults: Optional[dict] = None

    # Разрешаем произвольные дополнительные поля, т.к. структура может отличаться.
    model_config = {"extra": "allow"}


class TweetResults(BaseModel):
    """Результат твита может быть валидным (TweetResult) или «урезанным» (UnknownTweetResult)."""

    result: Union[TweetResult, UnknownTweetResult]


class ItemContent(BaseModel):
    """`content › itemContent`"""
    itemType: str = Field(..., alias="itemType")
    tweet_results: TweetResults

    model_config = {"populate_by_name": True}


class EntryContent(BaseModel):
    """`entries[n] › content`"""
    entryType: str = Field(..., alias="entryType")
    itemContent: Optional[ItemContent] = None

    model_config = {"populate_by_name": True}


class Entry(BaseModel):
    entryId: str
    sortIndex: str
    content: EntryContent


class Instruction(BaseModel):
    type: str
    entries: List[Entry]


class Timeline(BaseModel):
    instructions: List[Instruction]


class SearchTimeline(BaseModel):
    timeline: Timeline


class SearchByRawQuery(BaseModel):
    search_timeline: SearchTimeline


class DataLevel2(BaseModel):
    search_by_raw_query: SearchByRawQuery


class GraphQLResponse(BaseModel):
    """Top-level object of the decoded JSON"""
    data: DataLevel2


class FeedResponse(BaseModel):
    tweets: List[TweetResult]


class UserProfileLegacy(BaseModel):
    """Расширенная информация о пользователе из API"""
    screen_name: str
    name: str
    followers_count: int
    friends_count: int
    statuses_count: int
    profile_image_url_https: str
    created_at: str
    description: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    verified: bool = False


class UserProfileResult(BaseModel):
    """Результат пользователя из API"""
    rest_id: str
    legacy: UserProfileLegacy
    typename: str = Field(default="User", alias="__typename")

    model_config = {"populate_by_name": True, "extra": "allow"}


class UserProfileWrapper(BaseModel):
    """Обёртка для результата пользователя"""
    result: UserProfileResult


class UserProfileData(BaseModel):
    """Данные пользователя верхнего уровня"""
    user: UserProfileWrapper


class UserProfileDataWrapper(BaseModel):
    """Обёртка для данных пользователя"""
    data: UserProfileData


class UserProfileResponse(BaseModel):
    """Полный ответ API для профиля пользователя"""
    code: int
    data: UserProfileDataWrapper
    result: Optional[Any] = None
    msg: str


class XApiService:
    
    API_HOST = "twitter-api-v1-1-enterprise.p.rapidapi.com"
    
    def __init__(self):
        self.api_key = settings.X_RAPIDAPI_KEY
        self.base_url = self.API_HOST
        if not self.api_key:
            raise ValueError("X_RAPIDAPI_KEY environment variable is not set")

    async def search(self, query: str = "from:* ", from_timestamp: int = None) -> FeedResponse:
        """
        Поиск постов в X (Twitter) за весь промежуток времени от указанной даты до текущей
        
        Args:
            query: Поисковый запрос (по умолчанию возвращает все посты)
        
        Вызываем _get_feed всегда с указанным from_timestamp, 
        но since_timestamp всегда снижаем как минимальную дату в поиске.
        И так, пока не дойдем до текущей даты или посты не закончатся.
        """
        until_timestamp = None
        tweets = []
        logger.info(f"Searching for tweets from {from_timestamp} to {until_timestamp} with query: {query}")
        
        while True:
            # получаем ленту
            feed_response: GraphQLResponse = await self._get_serch_feed(query, from_timestamp, until_timestamp)
            
            # извлекаем твиты из ответа
            new_tweets, old_tweets_exist = self._extract_actual_tweets_from_feed(feed_response, from_timestamp)
            logger.info(f"Found {len(new_tweets)} new tweets")
            tweets.extend(new_tweets)

            # если хотябы один пост был выкинут или нет новых постов, то можно завершать цикл
            # по логике не должно быть более старых постов, чем в from_timestamp, но лучше проверять
            if old_tweets_exist or not new_tweets:
                break

            # снижаем until_timestamp на время самого старого поста
            until_timestamp = min(parse_date_to_timestamp(tweet.legacy.created_at) for tweet in new_tweets)
            logger.info(f"Until timestamp changed to: {until_timestamp}")
        
        logger.info(f"Found {len(tweets)} tweets")
        
        # чисто проверка на дубликаты
        tweets_ids = dict()
        for tweet in tweets:
            if tweet.rest_id in tweets_ids:
                logger.error(f"Duplicate tweet found: {tweet.rest_id}")
            else:
                tweets_ids[tweet.rest_id] = tweet
        
        return FeedResponse(tweets=tweets)
        
    async def _get_serch_feed(self, query:str, from_timestamp: int, until_timestamp: int = None) -> GraphQLResponse:
        """
        Получение ленты постов
        """
        headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.API_HOST
        }
        
        params = {
            'words': query,
            'since': timestamp_to_X_date(from_timestamp),
            'resFormat': 'json',
            'apiKey': settings.X_TWITTER_API_KEY
        }

        if until_timestamp:
            params['until'] = timestamp_to_X_date(until_timestamp)

        response_json = None
        
        # Используем контекстный менеджер для каждого запроса - просто и надежно
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
        ) as session:
            try:
                async with session.get(
                    f"https://{self.base_url}/base/apitools/search",
                    headers=headers,
                    params=params
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    logger.debug(f"Raw API response: {response_json}")
            except aiohttp.ClientError as e:
                raise Exception(f"Error making request to X API: {str(e)}")
            
        if not response_json:
            raise Exception("Empty response from API")
            
        if isinstance(response_json, str):
            try:
                response_json = json.loads(response_json)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse API response as JSON: {str(e)}")
            
        if not isinstance(response_json, dict):
            raise Exception(f"Unexpected response format: {type(response_json)}")
            
        if "data" in response_json and isinstance(response_json["data"], dict):
            response_json = response_json["data"]
        else:
            raise Exception(f"Unexpected data payload: {response_json}")
        
        try:
            data = GraphQLResponse(**response_json)
            return data
        except Exception as e:
            logger.error(f"Failed to parse response as GraphQLResponse: {str(e)}")
            logger.error(f"Response data: {response_json}")
            raise Exception(f"Failed to parse API response: {str(e)}")
    
    
    def _extract_actual_tweets_from_feed(self, feed_response: GraphQLResponse, from_timestamp: int) -> Tuple[List[TweetResult], bool]:
        """
        Извлечение актуальных постов из ленты
        """
        tweets = []
        old_tweets_exist = False
        
        try:
            entries: List[Entry] = feed_response.data.search_by_raw_query.search_timeline.timeline.instructions[0].entries
        except (AttributeError, IndexError) as e:
            logger.error(f"Error accessing entries in feed response: {e}")
            logger.error(f"Feed response structure: {feed_response}")
            return [], False
        
        for entry in entries:
            try:
                if entry.content.itemContent is None:
                    logger.warning(f"Entry {entry.entryId} has no itemContent")
                    continue
                
                if not hasattr(entry.content.itemContent, 'tweet_results') or entry.content.itemContent.tweet_results is None:
                    logger.warning(f"Entry {entry.entryId} has no tweet_results")
                    continue
                
                if not hasattr(entry.content.itemContent.tweet_results, 'result') or entry.content.itemContent.tweet_results.result is None:
                    logger.warning(f"Entry {entry.entryId} has no result in tweet_results")
                    continue
                
                tweet_raw = entry.content.itemContent.tweet_results.result

                # Нас интересуют только полноценные твиты, которые прошли валидацию TweetResult.
                if not isinstance(tweet_raw, TweetResult):
                    logger.debug(f"Entry {entry.entryId} contains non-tweet object ({type(tweet_raw).__name__}), skipping")
                    continue

                tweet_result: TweetResult = tweet_raw

                # Фильтруем по дате – пропускаем «старые» твиты
                if parse_date_to_timestamp(tweet_result.legacy.created_at) < from_timestamp:
                    old_tweets_exist = True
                    continue

                tweets.append(tweet_result)
                    
            except Exception as e:
                logger.error(f"Error processing entry {entry.entryId}: {e}")
                logger.error(f"Entry data: {entry}")
                continue
            
        return tweets, old_tweets_exist
    
    
    async def get_user_social_id(self, user_login: str) -> str:
        """
        Получение social_id пользователя по его логину
        
        Args:
            user_login: Логин пользователя в X (Twitter)
            
        Returns:
            str: rest_id пользователя
            
        Raises:
            Exception: Если не удалось получить или извлечь rest_id
        """
        headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.API_HOST
        }
        
        params = {
            'screenName': user_login,
            'resFormat': 'json',
            'apiKey': settings.X_TWITTER_API_KEY
        }

        response_json = None
        
        # Используем контекстный менеджер для каждого запроса
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
        ) as session:
            try:
                async with session.get(
                    f"https://{self.base_url}/base/apitools/userByScreenNameV2",
                    headers=headers,
                    params=params
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    logger.debug(f"User API response: {response_json}")
            except aiohttp.ClientError as e:
                raise Exception(f"Error making request to X API for user {user_login}: {str(e)}")
            
        if not response_json:
            raise Exception(f"Empty response from API for user {user_login}")
            
        if isinstance(response_json, str):
            try:
                response_json = json.loads(response_json)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse API response as JSON for user {user_login}: {str(e)}")
            
        if not isinstance(response_json, dict):
            raise Exception(f"Unexpected response format for user {user_login}: {type(response_json)}")
        
        # Пытаемся извлечь rest_id из разных возможных путей
        rest_id = None
        
        try:
            # Основной путь: data.data.user.result.rest_id
            if "data" in response_json and "data" in response_json["data"] and "user" in response_json["data"]["data"]:
                user_data = response_json["data"]["data"]["user"]
                if "result" in user_data and "rest_id" in user_data["result"]:
                    rest_id = user_data["result"]["rest_id"]
                    logger.info(f"Found rest_id for user {user_login}: {rest_id}")
                    return rest_id
        except (KeyError, TypeError) as e:
            logger.debug(f"Primary path failed for user {user_login}: {e}")
        
        # Альтернативные пути для rest_id
        alternative_paths = [
            ["data", "user", "result", "rest_id"],
            ["user", "result", "rest_id"],
            ["result", "rest_id"],
            ["rest_id"]
        ]
        
        for path in alternative_paths:
            try:
                current = response_json
                for key in path:
                    current = current[key]
                if current:
                    rest_id = str(current)
                    logger.info(f"Found rest_id for user {user_login} via alternative path {path}: {rest_id}")
                    return rest_id
            except (KeyError, TypeError):
                continue
        
        # Если не удалось найти rest_id, выбрасываем ошибку с подробной информацией
        logger.error(f"Failed to extract rest_id for user {user_login}")
        logger.error(f"Response structure: {json.dumps(response_json, indent=2)}")
        raise Exception(
            f"Could not extract rest_id for user '{user_login}'. "
            f"Response structure does not match expected format. "
            f"Available keys at root level: {list(response_json.keys()) if isinstance(response_json, dict) else 'Not a dict'}"
        )
    
    

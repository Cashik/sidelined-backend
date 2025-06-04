import os
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid
import aiohttp
from pydantic import BaseModel, Field
import logging
import json
import asyncio

from src.config.settings import settings
from src.utils_base import parse_date_to_timestamp, timestamp_to_X_date

logger = logging.getLogger(__name__)


class UserLegacy(BaseModel):
    name:          str
    screen_name:   str
    followers_count: int
    profile_image_url_https: str

    model_config = {"extra": "allow"}

class User(BaseModel):
    """`core › user_results › result`"""
    rest_id: str
    legacy: UserLegacy

    model_config = {"extra": "allow"}
    
class TweetLegacy(BaseModel):
    id_str: str
    full_text: str
    created_at: str
    favorite_count: int
    retweet_count: int
    reply_count: int
    lang: str
    
    # сюда подтягиваются поля из JSON для проверки reply/quote
    in_reply_to_status_id_str: Optional[str] = None
    in_reply_to_user_id_str: Optional[str] = None
    is_quote_status: bool = False

    model_config = {"extra": "allow"}
    
class TweetViews(BaseModel):
    count: Optional[str] = None
    state: str

    model_config = {"extra": "allow"}
    
class UserResults(BaseModel):
    """`core › user_results` wrapper"""
    result: User

    model_config = {"extra": "allow"}
    
class TweetCore(BaseModel):
    user_results: UserResults
    
    model_config = {"extra": "allow"}


class TweetResult(BaseModel):
    """`tweet_results › result`"""
    rest_id: str
    core: TweetCore
    legacy: TweetLegacy
    views: Optional[TweetViews] = None

    # Здесь добавляем поля из JSON, которые указывают на ретвит или цитату:
    retweeted_status_result: Optional["TweetResults"] = None
    quoted_status_result: Optional["TweetResults"] = None
    
    model_config = {"extra": "allow"}

    def is_retweet(self) -> bool:
        return self.retweeted_status_result is not None

    def is_reply(self) -> bool:
        # reply ⇔ legacy.in_reply_to_status_id_str не None
        return self.legacy.in_reply_to_status_id_str is not None

    def is_quote(self) -> bool:
        # quote ⇔ legacy.is_quote_status=True (часто вместе с наличием quoted_status_result)
        return self.legacy.is_quote_status

    def is_original(self) -> bool:
        return not (self.is_retweet() or self.is_reply() or self.is_quote())

class UnknownTweetResult(BaseModel):
    """Заглушка для твитов без обязательных полей (rest_id / core / legacy)."""

    limitedActionResults: Optional[dict] = None

    # Разрешаем произвольные дополнительные поля, т.к. структура может отличаться.
    model_config = {"extra": "allow"}


class TweetResults(BaseModel):
    """Результат твита может быть валидным (TweetResult) или «урезанным» (UnknownTweetResult)."""

    result: Union[TweetResult, UnknownTweetResult]

    model_config = {"extra": "allow"}
class ItemContent(BaseModel):
    """`content › itemContent`"""
    itemType: str = Field(..., alias="itemType")
    tweet_results: TweetResults

    model_config = {"populate_by_name": True, "extra": "allow"}


class EntryContent(BaseModel):
    """`entries[n] › content`"""
    entryType: str = Field(..., alias="entryType")
    itemContent: Optional[ItemContent] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


class Entry(BaseModel):
    entryId: str
    sortIndex: str
    content: EntryContent

    model_config = {"populate_by_name": True, "extra": "allow"}
    
class Instruction(BaseModel):
    type: str
    entries: List[Entry]

    model_config = {"populate_by_name": True, "extra": "allow"}
    
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

# После того как все классы объявлены, «связываем» forward ref:
TweetResult.model_rebuild()
TweetResults.model_rebuild()


class XApiService:
    
    API_HOST = "twitter-api-v1-1-enterprise.p.rapidapi.com"
    RETRAY_COUNT = 3
    RETRAY_DELAY = 5
    BASE_REQUEST_DELAY = 1
    
    def __init__(self):
        self.api_key = settings.X_RAPIDAPI_KEY
        self.base_url = self.API_HOST
        if not self.api_key:
            raise ValueError("X_RAPIDAPI_KEY environment variable is not set")

    def extract_tweet_results_from_feed(self, feed_json: dict) -> list:
        """
        Извлекает только TweetResults (твиты) из json-ответа X API.
        """
        tweet_results = []
        data = feed_json.get("data", {})
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        instructions = data.get("search_by_raw_query", {}).get("search_timeline", {}).get("timeline", {}).get("instructions", [])
        
        for instr in instructions:
            for entry in instr.get("entries", []):
                content = entry.get("content", {})
                if content.get("entryType") == "TimelineTimelineItem":
                    item_content = content.get("itemContent", {})
                    tweet_results_json = item_content.get("tweet_results")
                    if tweet_results_json:
                        try:
                            tweet_result = TweetResults(**tweet_results_json)
                            tweet_results.append(tweet_result)
                        except Exception as e:
                            logger.warning(f"Failed to parse TweetResults: {e}")
        return tweet_results

    async def _get_serch_feed(self, query:str, cursor: Optional[str] = None) -> Tuple[list, Optional[str]]:
        """
        Получение твитов (TweetResults) + извлечение курсора Bottom
        """
        headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.API_HOST
        }
        params = {
            'words': query,
            'resFormat': 'json',
            'apiKey': settings.X_TWITTER_API_KEY,
        }
        if cursor:
            params['cursor'] = cursor
        response_json = None
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
                    logger.info(f"Raw API response: {response_json}")
            except aiohttp.ClientError as e:
                raise Exception(f"Error making request to X API: {str(e)}")
        if not response_json:
            raise Exception("Empty response from API")
        if isinstance(response_json, str):
            try:
                response_json = json.loads(response_json)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse API response as JSON: {str(e)} \n Response: {response_json}")
        if not isinstance(response_json, dict):
            raise Exception(f"Unexpected response format: {type(response_json)}")
        # Извлекаем твиты и курсор из исходного json
        tweet_results = self.extract_tweet_results_from_feed(response_json)
        bottom_cursor = self.extract_bottom_cursor_from_feed(response_json)
        return tweet_results, bottom_cursor

    async def _get_serch_feed_with_retry(self, query: str, cursor: Optional[str] = None) -> Tuple[list, Optional[str]]:
        """
        Обёртка над _get_serch_feed с retry-логикой и задержками.
        """
        last_exception = None
        for attempt in range(1, self.RETRAY_COUNT + 1):
            try:
                result = await self._get_serch_feed(query, cursor)
                if attempt > 1:
                    logger.info(f"_get_serch_feed succeeded on retry {attempt}")
                return result
            except Exception as e:
                last_exception = e
                logger.warning(f"_get_serch_feed failed (attempt {attempt}/{self.RETRAY_COUNT}): {e}")
                await asyncio.sleep(self.RETRAY_DELAY)
        logger.error(f"_get_serch_feed failed after {self.RETRAY_COUNT} attempts: {last_exception}")
        raise last_exception

    async def search(self, query: str = "from:* ", from_timestamp: int = None, min_likes_count: int = None) -> FeedResponse:
        """
        Поиск постов в X (Twitter) начиная с from_timestamp (UNIX-время).
        Использует курсор для пагинации, пока не встретит твиты старше from_timestamp или не закончится курсор.
        Стабильная версия: retry, задержки, возврат частичных данных при ошибках.
        """
        tweets = []
        seen_ids = set()
        cursor = None
        base_query = f"{query} since:{timestamp_to_X_date(from_timestamp)} lang:en"
        logger.info(f"Searching for tweets from {from_timestamp} with query: {base_query}")
        had_success = False
        while True:
            try:
                tweet_results, next_cursor = await self._get_serch_feed_with_retry(base_query, cursor)
            except Exception as e:
                if tweets:
                    logger.error(f"Search stopped due to error after partial success: {e}")
                    break
                else:
                    logger.error(f"Search failed with no data: {e}")
                    raise
            logger.info(f"Tweet results count: {len(tweet_results)}")
            new_tweets = []
            for tweet_result in tweet_results:
                tweet = tweet_result.result
                if isinstance(tweet, TweetResult):
                    logger.info(f"Tweet: {tweet}")
                    if tweet.rest_id not in seen_ids:
                        if parse_date_to_timestamp(tweet.legacy.created_at) >= from_timestamp:
                            new_tweets.append(tweet)
                            seen_ids.add(tweet.rest_id)
                    else:
                        logger.info(f"Tweet already seen: {tweet.rest_id}")
            tweets.extend(new_tweets)
            logger.info(f"Fetched {len(new_tweets)} new tweets, total: {len(tweets)}")
            if not new_tweets:
                logger.info("No new tweets found or all are duplicates/too old. Stopping.")
                break
            if not next_cursor:
                logger.info("No next cursor. End of feed.")
                break
            cursor = next_cursor
            await asyncio.sleep(self.BASE_REQUEST_DELAY)
        return FeedResponse(tweets=tweets)
        
    def extract_bottom_cursor_from_feed(self, feed_json: dict) -> Optional[str]:
        """
        Извлекает значение курсора Bottom из json-ответа X API (ищет во всех entries всех instructions).
        """
        try:
            data = feed_json.get("data", {})
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            raw_query = data.get("search_by_raw_query", {})
            search_timeline = raw_query.get("search_timeline", {})
            timeline = search_timeline.get("timeline", {})
            instructions = timeline.get("instructions", [])
            logger.info(f"Instructions: {str(instructions)[:100]}")
            for instr in instructions:
                entries = instr.get("entries", [])
                for entry in entries:
                    content = entry.get("content", {})
                    # Иногда курсор лежит прямо в entry.content
                    if (
                        content.get("entryType") == "TimelineTimelineCursor"
                        and content.get("cursorType") == "Bottom"
                        and "value" in content
                    ):
                        logger.info(f"Found bottom cursor in entry.content: {content['value']}")
                        return content["value"]
            # Иногда курсор лежит прямо в entries (без content)
            for instr in instructions:
                entry = instr.get("entry", {})
                entry_content = entry.get("content", {})
                if entry_content.get("cursorType") == "Bottom" and "value" in entry_content:
                    logger.info(f"Found bottom cursor in entries: {entry_content['value']}")
                    return entry_content["value"]
        except Exception as e:
            logger.warning(f"Failed to extract bottom cursor: {e}")
        return None

    def _extract_actual_tweets_from_feed(self, feed_response: GraphQLResponse, from_timestamp: int) -> Tuple[List[TweetResult], bool]:
        """
        Извлечение актуальных постов из ленты
        """
        tweets = []
        old_tweets_exist = False
        
        try:
            entries: List[Entry] = feed_response.data.search_by_raw_query.search_timeline.timeline.instructions[0].entries
        except (AttributeError, IndexError) as e:
            logger.error(f"Error accessing entries in feed response: {e} \n Feed response: {feed_response}")
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
                logger.error(f"Error processing entry {entry.entryId}: {e} \n Entry: {entry}")
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
    
    

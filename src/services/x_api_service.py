import os
from typing import Dict, Any, List, Optional, Tuple
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


class TweetResults(BaseModel):
    result: TweetResult


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
        # TODO: придумать, что делать в разных случаях ошибок
        
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
        async with aiohttp.ClientSession() as session:
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
            
        if "data" in response_json:
            response_json = response_json["data"]
        else:
            raise Exception("No data field in response")
        
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
                
                tweet_result: TweetResult = entry.content.itemContent.tweet_results.result
                
                if parse_date_to_timestamp(tweet_result.legacy.created_at) < from_timestamp:
                    old_tweets_exist = True
                    continue
                else:
                    tweets.append(tweet_result)
                    
            except Exception as e:
                logger.error(f"Error processing entry {entry.entryId}: {e}")
                logger.error(f"Entry data: {entry}")
                continue
            
        return tweets, old_tweets_exist

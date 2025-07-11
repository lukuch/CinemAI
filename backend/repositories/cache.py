import redis
from domain.interfaces import CacheRepository
from core.settings import settings

class RedisCacheRepository(CacheRepository):
    def __init__(self):
        self.redis = redis.from_url(settings.redis_url)
    def get(self, key: str):
        return self.redis.get(key)
    def set(self, key: str, value, expire: int = 3600):
        self.redis.set(key, value, ex=expire) 
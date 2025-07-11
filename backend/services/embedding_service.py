import openai
from domain.interfaces import EmbeddingService
from domain.entities import Embedding
from core.settings import settings
from typing import List
import asyncio
import redis
import hashlib
import json

class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.redis = redis.from_url(settings.redis_url)
        self.model = "text-embedding-3-large"

    def _cache_key(self, text: str) -> str:
        return f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"

    def embed(self, texts: List[str]) -> List[Embedding]:
        results = []
        uncached = []
        uncached_indices = []
        # Check cache first
        for i, text in enumerate(texts):
            cached = self.redis.get(self._cache_key(text))
            if cached:
                results.append(Embedding(vector=json.loads(cached)))
            else:
                results.append(None)
                uncached.append(text)
                uncached_indices.append(i)
        # Batch uncached
        if uncached:
            response = self.client.embeddings.create(
                input=uncached,
                model=self.model
            )
            for idx, emb in zip(uncached_indices, response.data):
                vector = emb.embedding
                results[idx] = Embedding(vector=vector)
                self.redis.set(self._cache_key(uncached[idx-uncached_indices[0]]), json.dumps(vector), ex=60*60*24*30)
        return results 
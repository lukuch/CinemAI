import asyncio
import hashlib
import json
from typing import List

import openai
import redis

from core.settings import settings
from domain.entities import Embedding
from domain.interfaces import EmbeddingService


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.redis = redis.from_url(settings.redis_url)
        self.model = "text-embedding-3-large"

    def _cache_key(self, text: str) -> str:
        return f"embedding:{hashlib.sha256(text.encode()).hexdigest()}"

    async def embed_batch(self, batch_texts, batch_indices, results):
        response = await self.client.embeddings.create(input=batch_texts, model=self.model)
        for j, emb in enumerate(response.data):
            idx = batch_indices[j]
            vector = emb.embedding
            results[idx] = Embedding(vector=vector)
            self.redis.set(self._cache_key(batch_texts[j]), json.dumps(vector), ex=60 * 60 * 24 * 30)

    async def embed(self, texts: List[str]) -> List[Embedding]:
        keys = [self._cache_key(text) for text in texts]
        cached_values = self.redis.mget(keys)
        results = []
        uncached = []
        uncached_indices = []
        for i, cached in enumerate(cached_values):
            if cached:
                results.append(Embedding(vector=json.loads(cached)))
            else:
                results.append(None)
                uncached.append(texts[i])
                uncached_indices.append(i)

        # Parallelize batches
        batch_size = 100
        tasks = []
        for batch_start in range(0, len(uncached), batch_size):
            batch_texts = uncached[batch_start : batch_start + batch_size]
            batch_indices = uncached_indices[batch_start : batch_start + batch_size]
            tasks.append(self.embed_batch(batch_texts, batch_indices, results))
        if tasks:
            await asyncio.gather(*tasks)
        return results

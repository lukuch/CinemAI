import asyncio
import hashlib
from typing import Dict, List, Optional, Tuple

import openai
import orjson
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

    async def _embed_batch(self, batch_texts, batch_indices, results):
        try:
            response = await self.client.embeddings.create(input=batch_texts, model=self.model)
            pipe = self.redis.pipeline()
            for j, emb in enumerate(response.data):
                idx = batch_indices[j]
                vector = emb.embedding
                results[idx] = Embedding(vector=vector)
                pipe.set(self._cache_key(batch_texts[j]), orjson.dumps(vector), ex=60 * 60 * 24 * 30)
            pipe.execute()
        except Exception:
            for j in range(len(batch_texts)):
                idx = batch_indices[j]
                results[idx] = None

    def _deduplicate_texts(self, texts: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
        text_to_indices: Dict[str, List[int]] = {}
        for i, text in enumerate(texts):
            text_to_indices.setdefault(text, []).append(i)
        unique_texts = list(text_to_indices.keys())
        return unique_texts, text_to_indices

    def _get_cached_embeddings(self, unique_texts: List[str]) -> List[Optional[Embedding]]:
        keys = [self._cache_key(text) for text in unique_texts]
        cached_values = self.redis.mget(keys)
        unique_results = [Embedding(vector=orjson.loads(cached)) if cached else None for cached in cached_values]
        return unique_results

    def _get_uncached_batches(
        self, unique_texts: List[str], unique_results: List[Optional[Embedding]], batch_size: int = 100
    ) -> List[Tuple[List[str], List[int]]]:
        uncached = []
        uncached_indices = []
        for i, result in enumerate(unique_results):
            if result is None:
                uncached.append(unique_texts[i])
                uncached_indices.append(i)
        batches = []
        for batch_start in range(0, len(uncached), batch_size):
            batch_texts = uncached[batch_start : batch_start + batch_size]
            batch_indices = uncached_indices[batch_start : batch_start + batch_size]
            batches.append((batch_texts, batch_indices))
        return batches

    def _map_results_to_original_order(
        self,
        text_to_indices: Dict[str, List[int]],
        unique_texts: List[str],
        unique_results: List[Optional[Embedding]],
        total: int,
    ) -> List[Embedding]:
        results = [None] * total
        for text, indices in text_to_indices.items():
            unique_idx = unique_texts.index(text)
            for i in indices:
                results[i] = unique_results[unique_idx]
        return results

    async def embed(self, texts: List[str]) -> List[Embedding]:
        unique_texts, text_to_indices = self._deduplicate_texts(texts)
        unique_results = self._get_cached_embeddings(unique_texts)
        batches = self._get_uncached_batches(unique_texts, unique_results)
        tasks = [self._embed_batch(batch_texts, batch_indices, unique_results) for batch_texts, batch_indices in batches]
        if tasks:
            await asyncio.gather(*tasks)
        return self._map_results_to_original_order(text_to_indices, unique_texts, unique_results, len(texts))

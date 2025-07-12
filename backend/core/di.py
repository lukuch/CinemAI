from injector import Injector, singleton
from fastapi import FastAPI
from services.embedding_service import OpenAIEmbeddingService
from services.clustering_service import SklearnClusteringService
from services.tmdb_service import TMDBApiService
from services.filtering_service import DefaultFilteringService
from services.recommendation_service import DefaultRecommendationService
from services.llm_service import OpenAILLMService
from repositories.cache import RedisCacheRepository
from repositories.vector_store import PgvectorRepository

def create_injector() -> Injector:
    injector = Injector()
    injector.binder.bind(OpenAIEmbeddingService, to=OpenAIEmbeddingService, scope=singleton)
    injector.binder.bind(SklearnClusteringService, to=SklearnClusteringService, scope=singleton)
    injector.binder.bind(TMDBApiService, to=TMDBApiService, scope=singleton)
    injector.binder.bind(DefaultFilteringService, to=DefaultFilteringService, scope=singleton)
    injector.binder.bind(DefaultRecommendationService, to=DefaultRecommendationService, scope=singleton)
    injector.binder.bind(OpenAILLMService, to=OpenAILLMService, scope=singleton)
    injector.binder.bind(RedisCacheRepository, to=RedisCacheRepository, scope=singleton)
    return injector 
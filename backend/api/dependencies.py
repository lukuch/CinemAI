from fastapi import Depends
from fastapi_injector import Injected
from sqlalchemy.ext.asyncio import AsyncSession
from db.deps import get_async_session
from managers.recommendation_manager import RecommendationManager
from services.embedding_service import OpenAIEmbeddingService
from services.clustering_service import SklearnClusteringService
from services.tmdb_service import TMDBApiService
from services.filtering_service import DefaultFilteringService
from services.recommendation_service import DefaultRecommendationService
from services.llm_service import OpenAILLMService


def get_recommendation_manager(
    session: AsyncSession = Depends(get_async_session),
    embedder: OpenAIEmbeddingService = Injected(OpenAIEmbeddingService),
    clusterer: SklearnClusteringService = Injected(SklearnClusteringService),
    tmdb: TMDBApiService = Injected(TMDBApiService),
    filterer: DefaultFilteringService = Injected(DefaultFilteringService),
    recommender: DefaultRecommendationService = Injected(DefaultRecommendationService),
    llm: OpenAILLMService = Injected(OpenAILLMService),
) -> RecommendationManager:
    return RecommendationManager(
        embedder=embedder,
        clusterer=clusterer,
        tmdb=tmdb,
        filterer=filterer,
        recommender=recommender,
        llm=llm,
        session=session,
    ) 
from fastapi import Depends
from fastapi_injector import Injected
from sqlalchemy.ext.asyncio import AsyncSession
from structlog.stdlib import BoundLogger

from db.deps import get_async_session
from domain.interfaces import FieldDetectionService
from managers.recommendation_manager import RecommendationManager
from repositories.vector_store import PgvectorRepository
from services.clustering_service import SklearnClusteringService
from services.embedding_service import OpenAIEmbeddingService
from services.field_detection_service import DefaultFieldDetectionService
from services.filtering_service import DefaultFilteringService
from services.llm_service import OpenAILLMService
from services.recommendation_service import DefaultRecommendationService
from services.tmdb_service import TMDBApiService


def get_recommendation_manager(
    session: AsyncSession = Depends(get_async_session),
    embedder: OpenAIEmbeddingService = Injected(OpenAIEmbeddingService),
    clusterer: SklearnClusteringService = Injected(SklearnClusteringService),
    tmdb: TMDBApiService = Injected(TMDBApiService),
    filterer: DefaultFilteringService = Injected(DefaultFilteringService),
    recommender: DefaultRecommendationService = Injected(DefaultRecommendationService),
    field_detector: FieldDetectionService = Injected(DefaultFieldDetectionService),
    llm: OpenAILLMService = Injected(OpenAILLMService),
    logger: BoundLogger = Injected(BoundLogger),
) -> RecommendationManager:
    return RecommendationManager(
        embedder=embedder,
        clusterer=clusterer,
        tmdb=tmdb,
        filterer=filterer,
        recommender=recommender,
        field_detector=field_detector,
        llm=llm,
        vectorstore=PgvectorRepository(session),
        logger=logger,
    )

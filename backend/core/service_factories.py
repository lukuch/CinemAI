from fastapi import Depends
from fastapi_injector import Injected
from sqlalchemy.ext.asyncio import AsyncSession
from structlog.stdlib import BoundLogger

from db.deps import get_async_session
from domain.interfaces import (
    IClusteringService,
    IEmbeddingService,
    IFieldDetectionService,
    IFilteringService,
    ILLMService,
    IMovieApiService,
    IRecommendationService,
    IUserProfileService,
    IVisualizationService,
)
from managers.recommendation_manager import RecommendationManager
from repositories.vector_store import PgvectorRepository
from services.user_profile_service import UserProfileService
from services.visualization_service import VisualizationService


def get_recommendation_manager(
    session: AsyncSession = Depends(get_async_session),
    embedder: IEmbeddingService = Injected(IEmbeddingService),
    clusterer: IClusteringService = Injected(IClusteringService),
    tmdb: IMovieApiService = Injected(IMovieApiService),
    filterer: IFilteringService = Injected(IFilteringService),
    recommender: IRecommendationService = Injected(IRecommendationService),
    field_detector: IFieldDetectionService = Injected(IFieldDetectionService),
    llm: ILLMService = Injected(ILLMService),
    logger: BoundLogger = Injected(BoundLogger),
) -> RecommendationManager:
    user_profile_service = get_user_profile_service(
        session=session,
        embedder=embedder,
        clusterer=clusterer,
        tmdb=tmdb,
        field_detector=field_detector,
        logger=logger,
    )

    return RecommendationManager(
        embedder=embedder,
        clusterer=clusterer,
        tmdb=tmdb,
        filterer=filterer,
        recommender=recommender,
        field_detector=field_detector,
        llm=llm,
        vectorstore=PgvectorRepository(session),
        user_profile_service=user_profile_service,
        logger=logger,
    )


def get_user_profile_service(
    session: AsyncSession = Depends(get_async_session),
    embedder: IEmbeddingService = Injected(IEmbeddingService),
    clusterer: IClusteringService = Injected(IClusteringService),
    tmdb: IMovieApiService = Injected(IMovieApiService),
    field_detector: IFieldDetectionService = Injected(IFieldDetectionService),
    logger: BoundLogger = Injected(BoundLogger),
) -> IUserProfileService:
    return UserProfileService(
        embedder=embedder,
        clusterer=clusterer,
        tmdb=tmdb,
        field_detector=field_detector,
        vectorstore=PgvectorRepository(session),
        logger=logger,
    )


def get_visualization_service(
    service: IVisualizationService = Injected(VisualizationService),
) -> IVisualizationService:
    return service

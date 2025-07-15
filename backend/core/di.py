import structlog
from injector import Injector, singleton
from structlog.stdlib import BoundLogger

from domain.interfaces import (
    ICacheRepository,
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
from repositories.cache import RedisCacheRepository
from services.clustering_service import SklearnClusteringService
from services.embedding_service import OpenAIEmbeddingService
from services.field_detection_service import FieldDetectionService
from services.filtering_service import FilteringService
from services.llm_service import OpenAILLMService
from services.recommendation_service import RecommendationService
from services.tmdb_service import TMDBApiService
from services.user_profile_service import UserProfileService
from services.visualization_service import VisualizationService


def create_injector() -> Injector:
    injector = Injector()
    injector.binder.bind(IMovieApiService, to=TMDBApiService, scope=singleton)
    injector.binder.bind(
        IFieldDetectionService, to=FieldDetectionService, scope=singleton
    )
    injector.binder.bind(ICacheRepository, to=RedisCacheRepository, scope=singleton)
    injector.binder.bind(IUserProfileService, to=UserProfileService, scope=singleton)
    injector.binder.bind(IEmbeddingService, to=OpenAIEmbeddingService, scope=singleton)
    injector.binder.bind(
        IClusteringService, to=SklearnClusteringService, scope=singleton
    )
    injector.binder.bind(IFilteringService, to=FilteringService, scope=singleton)
    injector.binder.bind(
        IRecommendationService, to=RecommendationService, scope=singleton
    )
    injector.binder.bind(ILLMService, to=OpenAILLMService, scope=singleton)
    injector.binder.bind(
        BoundLogger, to=structlog.get_logger("cinemai"), scope=singleton
    )
    injector.binder.bind(
        IVisualizationService, to=VisualizationService, scope=singleton
    )
    return injector

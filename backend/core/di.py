import structlog
from injector import Injector, singleton
from structlog.stdlib import BoundLogger

from domain.interfaces import ClusteringService, EmbeddingService, FieldDetectionService, TMDBService
from repositories.cache import RedisCacheRepository
from services.clustering_service import SklearnClusteringService
from services.embedding_service import OpenAIEmbeddingService
from services.field_detection_service import DefaultFieldDetectionService
from services.filtering_service import DefaultFilteringService
from services.llm_service import OpenAILLMService
from services.recommendation_service import DefaultRecommendationService
from services.tmdb_service import TMDBApiService
from services.user_profile_service import UserProfileService


def create_injector() -> Injector:
    injector = Injector()
    injector.binder.bind(OpenAIEmbeddingService, to=OpenAIEmbeddingService, scope=singleton)
    injector.binder.bind(SklearnClusteringService, to=SklearnClusteringService, scope=singleton)
    injector.binder.bind(TMDBService, to=TMDBApiService, scope=singleton)
    injector.binder.bind(DefaultFilteringService, to=DefaultFilteringService, scope=singleton)
    injector.binder.bind(DefaultRecommendationService, to=DefaultRecommendationService, scope=singleton)
    injector.binder.bind(FieldDetectionService, to=DefaultFieldDetectionService, scope=singleton)
    injector.binder.bind(OpenAILLMService, to=OpenAILLMService, scope=singleton)
    injector.binder.bind(RedisCacheRepository, to=RedisCacheRepository, scope=singleton)
    injector.binder.bind(UserProfileService, to=UserProfileService, scope=singleton)
    injector.binder.bind(EmbeddingService, to=OpenAIEmbeddingService, scope=singleton)
    injector.binder.bind(ClusteringService, to=SklearnClusteringService, scope=singleton)
    injector.binder.bind(BoundLogger, to=structlog.get_logger("cinemai"), scope=singleton)
    return injector

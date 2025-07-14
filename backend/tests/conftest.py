import asyncio
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from injector import Injector
from structlog.stdlib import BoundLogger

from core.di import create_injector
from domain.entities import Cluster, Embedding, Movie, MovieHistoryItem, UserProfile
from domain.interfaces import (
    ClusteringService,
    EmbeddingService,
    FieldDetectionService,
    FilteringService,
    LLMService,
    RecommendationService,
    TMDBService,
    VectorStoreRepository,
)
from managers.recommendation_manager import RecommendationManager
from services.user_profile_service import UserProfileService


@pytest.fixture(autouse=True, scope="session")
def patch_openai_llm():
    """Automatically patch the LLM service rerank method for all tests to prevent real LLM calls."""
    with patch("services.llm_service.OpenAILLMService.rerank") as mock_rerank:

        mock_rerank.return_value = [
            {
                "title": "Interstellar",
                "year": 2014,
                "genres": ["Adventure", "Drama", "Sci-Fi"],
                "justification": "Because you like sci-fi movies.",
            },
            {
                "title": "The Dark Knight",
                "year": 2008,
                "genres": ["Action", "Crime", "Drama"],
                "justification": "Because you like action movies.",
            },
        ]
        yield


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def injector() -> Injector:
    """Create a test injector with mocked external dependencies."""
    return create_injector()


@pytest.fixture
def mock_logger() -> BoundLogger:
    """Create a mock logger for testing."""
    logger = MagicMock(spec=BoundLogger)
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    """Create a mock embedding service."""
    service = MagicMock(spec=EmbeddingService)

    test_embeddings = [
        Embedding([0.1, 0.2, 0.3, 0.4, 0.5] * 100),
        Embedding([0.2, 0.3, 0.4, 0.5, 0.6] * 100),
        Embedding([0.3, 0.4, 0.5, 0.6, 0.7] * 100),
        Embedding([0.4, 0.5, 0.6, 0.7, 0.8] * 100),
        Embedding([0.5, 0.6, 0.7, 0.8, 0.9] * 100),
    ]
    service.embed = AsyncMock(return_value=test_embeddings)
    return service


@pytest.fixture
def mock_clustering_service() -> ClusteringService:
    """Create a mock clustering service."""
    service = MagicMock(spec=ClusteringService)

    test_clusters = [
        Cluster(
            centroid=Embedding([0.15, 0.25, 0.35, 0.45, 0.55] * 100),
            movies=[
                Movie(
                    id="1",
                    title="The Matrix",
                    year=1999,
                    duration=136,
                    genres=["Action", "Sci-Fi"],
                    countries=["USA"],
                    description="A computer hacker learns from mysterious rebels about the true nature of his reality.",
                    embedding=Embedding([0.1, 0.2, 0.3, 0.4, 0.5] * 100),
                ),
                Movie(
                    id="2",
                    title="Inception",
                    year=2010,
                    duration=148,
                    genres=["Action", "Sci-Fi", "Thriller"],
                    countries=["USA", "UK"],
                    description="A thief who steals corporate secrets through dream-sharing technology.",
                    embedding=Embedding([0.2, 0.3, 0.4, 0.5, 0.6] * 100),
                ),
            ],
            average_rating=4.6,
            count=2,
        ),
    ]
    service.cluster.return_value = test_clusters
    return service


@pytest.fixture
def mock_tmdb_service() -> TMDBService:
    """Create a mock TMDB service."""
    service = AsyncMock(spec=TMDBService)

    test_candidates = [
        Movie(
            id="101",
            title="Interstellar",
            year=2014,
            duration=169,
            genres=["Adventure", "Drama", "Sci-Fi"],
            countries=["USA", "UK"],
            description="A team of explorers travel through a wormhole in space.",
            embedding=None,
        ),
        Movie(
            id="102",
            title="The Dark Knight",
            year=2008,
            duration=152,
            genres=["Action", "Crime", "Drama"],
            countries=["USA", "UK"],
            description="When the menace known as the Joker wreaks havoc on Gotham City.",
            embedding=None,
        ),
        Movie(
            id="103",
            title="Pulp Fiction",
            year=1994,
            duration=154,
            genres=["Crime", "Drama"],
            countries=["USA"],
            description="The lives of two mob hitmen, a boxer, a gangster and his wife intertwine.",
            embedding=None,
        ),
    ]

    service.fetch_movies = AsyncMock(return_value=test_candidates)
    return service


@pytest.fixture
def mock_filtering_service() -> FilteringService:
    """Create a mock filtering service."""
    service = MagicMock(spec=FilteringService)

    def mock_filter(movies, filters):
        return movies

    service.filter = mock_filter
    return service


@pytest.fixture
def mock_recommendation_service() -> RecommendationService:
    """Create a mock recommendation service."""
    service = MagicMock(spec=RecommendationService)

    test_recommendations = [
        (
            0.85,
            Movie(
                id="101",
                title="Interstellar",
                year=2014,
                duration=169,
                genres=["Adventure", "Drama", "Sci-Fi"],
                countries=["USA", "UK"],
                description="A team of explorers travel through a wormhole in space.",
                embedding=Embedding([0.15, 0.25, 0.35, 0.45, 0.55] * 100),
            ),
        ),
        (
            0.78,
            Movie(
                id="102",
                title="The Dark Knight",
                year=2008,
                duration=152,
                genres=["Action", "Crime", "Drama"],
                countries=["USA", "UK"],
                description="When the menace known as the Joker wreaks havoc on Gotham City.",
                embedding=Embedding([0.25, 0.35, 0.45, 0.55, 0.65] * 100),
            ),
        ),
    ]

    service.recommend.return_value = test_recommendations
    return service


@pytest.fixture
def mock_field_detection_service() -> FieldDetectionService:
    """Create a mock field detection service."""
    service = AsyncMock(spec=FieldDetectionService)

    test_converted = [
        {
            "title": "The Matrix",
            "year": 1999,
            "duration": 136,
            "genres": ["Action", "Sci-Fi"],
            "countries": ["USA"],
            "description": "A computer hacker learns from mysterious rebels about the true nature of his reality.",
            "rating": 4.5,
            "watched_at": "2023-01-15",
        },
        {
            "title": "Inception",
            "year": 2010,
            "duration": 148,
            "genres": ["Action", "Sci-Fi", "Thriller"],
            "countries": ["USA", "UK"],
            "description": "A thief who steals corporate secrets through dream-sharing technology.",
            "rating": 4.7,
            "watched_at": "2023-02-20",
        },
    ]

    service.convert_movies_batch.return_value = (test_converted, [])
    service.validate_movie_data.return_value = True
    return service


@pytest.fixture
def mock_llm_service() -> LLMService:
    """Create a mock LLM service."""
    service = MagicMock(spec=LLMService)

    test_reranked = [
        {
            "title": "Interstellar",
            "year": 2014,
            "genres": ["Adventure", "Drama", "Sci-Fi"],
            "justification": "Based on your love for sci-fi films like The Matrix and Inception, Interstellar offers a similar blend of mind-bending concepts and emotional depth.",
        },
        {
            "title": "The Dark Knight",
            "year": 2008,
            "genres": ["Action", "Crime", "Drama"],
            "justification": "Given your appreciation for complex narratives and action films, The Dark Knight provides a perfect balance of thrilling action and psychological depth.",
        },
    ]

    service.rerank.return_value = test_reranked
    return service


@pytest.fixture
def mock_vector_store_repository() -> VectorStoreRepository:
    """Create a mock vector store repository."""
    repository = AsyncMock(spec=VectorStoreRepository)

    test_profile = UserProfile(
        user_id="test_user",
        movies=[
            MovieHistoryItem(
                title="The Matrix",
                year=1999,
                duration=136,
                genres=["Action", "Sci-Fi"],
                countries=["USA"],
                description="A computer hacker learns from mysterious rebels about the true nature of his reality.",
                rating=4.5,
                watched_at="2023-01-15",
            ),
            MovieHistoryItem(
                title="Inception",
                year=2010,
                duration=148,
                genres=["Action", "Sci-Fi", "Thriller"],
                countries=["USA", "UK"],
                description="A thief who steals corporate secrets through dream-sharing technology.",
                rating=4.7,
                watched_at="2023-02-20",
            ),
        ],
        clusters=[
            Cluster(
                centroid=Embedding([0.15, 0.25, 0.35, 0.45, 0.55] * 100),
                movies=[
                    Movie(
                        id="1",
                        title="The Matrix",
                        year=1999,
                        duration=136,
                        genres=["Action", "Sci-Fi"],
                        countries=["USA"],
                        description="A computer hacker learns from mysterious rebels about the true nature of his reality.",
                        embedding=Embedding([0.1, 0.2, 0.3, 0.4, 0.5] * 100),
                    ),
                ],
                average_rating=4.5,
                count=1,
            ),
        ],
    )
    repository.get_user_profile = AsyncMock(return_value=test_profile)
    repository.save_user_profile = AsyncMock(return_value=None)
    return repository


@pytest.fixture
def mock_user_profile_service(
    mock_embedding_service: EmbeddingService,
    mock_clustering_service: ClusteringService,
    mock_tmdb_service: TMDBService,
    mock_field_detection_service: FieldDetectionService,
    mock_vector_store_repository: VectorStoreRepository,
    mock_logger: BoundLogger,
) -> UserProfileService:
    """Create a mock user profile service with all dependencies."""
    service = UserProfileService(
        embedder=mock_embedding_service,
        clusterer=mock_clustering_service,
        tmdb=mock_tmdb_service,
        field_detector=mock_field_detection_service,
        vectorstore=mock_vector_store_repository,
        logger=mock_logger,
    )
    return service


@pytest.fixture
def recommendation_manager(
    mock_embedding_service: EmbeddingService,
    mock_clustering_service: ClusteringService,
    mock_tmdb_service: TMDBService,
    mock_filtering_service: FilteringService,
    mock_recommendation_service: RecommendationService,
    mock_field_detection_service: FieldDetectionService,
    mock_llm_service: LLMService,
    mock_vector_store_repository: VectorStoreRepository,
    mock_logger: BoundLogger,
) -> RecommendationManager:
    """Create a recommendation manager with all mocked dependencies."""
    mock_user_profile_service = AsyncMock(spec=UserProfileService)

    from domain.entities import Cluster, Embedding, Movie, MovieHistoryItem, UserProfile

    test_profile = UserProfile(
        user_id="test_user",
        movies=[
            MovieHistoryItem(
                title="The Matrix",
                year=1999,
                duration=136,
                genres=["Action", "Sci-Fi"],
                countries=["USA"],
                description="A computer hacker learns from mysterious rebels about the true nature of his reality.",
                rating=4.5,
                watched_at="2023-01-15",
            ),
            MovieHistoryItem(
                title="Inception",
                year=2010,
                duration=148,
                genres=["Action", "Sci-Fi", "Thriller"],
                countries=["USA", "UK"],
                description="A thief who steals corporate secrets through dream-sharing technology.",
                rating=4.7,
                watched_at="2023-02-20",
            ),
        ],
        clusters=[
            Cluster(
                centroid=Embedding([0.15, 0.25, 0.35, 0.45, 0.55] * 100),
                movies=[
                    Movie(
                        id="1",
                        title="The Matrix",
                        year=1999,
                        duration=136,
                        genres=["Action", "Sci-Fi"],
                        countries=["USA"],
                        description="A computer hacker learns from mysterious rebels about the true nature of his reality.",
                        embedding=Embedding([0.1, 0.2, 0.3, 0.4, 0.5] * 100),
                    ),
                ],
                average_rating=4.5,
                count=1,
            ),
        ],
    )
    mock_user_profile_service.get_profile.return_value = test_profile

    return RecommendationManager(
        embedder=mock_embedding_service,
        clusterer=mock_clustering_service,
        tmdb=mock_tmdb_service,
        filterer=mock_filtering_service,
        recommender=mock_recommendation_service,
        field_detector=mock_field_detection_service,
        llm=mock_llm_service,
        vectorstore=mock_vector_store_repository,
        user_profile_service=mock_user_profile_service,
        logger=mock_logger,
    )


@pytest.fixture
def sample_movies_data() -> list:
    """Sample movies data for testing."""
    return [
        {
            "title": "The Matrix",
            "year": 1999,
            "duration": 136,
            "genres": ["Action", "Sci-Fi"],
            "countries": ["USA"],
            "description": "A computer hacker learns from mysterious rebels about the true nature of his reality.",
            "rating": 4.5,
            "watched_at": "2023-01-15",
        },
        {
            "title": "Inception",
            "year": 2010,
            "duration": 148,
            "genres": ["Action", "Sci-Fi", "Thriller"],
            "countries": ["USA", "UK"],
            "description": "A thief who steals corporate secrets through dream-sharing technology.",
            "rating": 4.7,
            "watched_at": "2023-02-20",
        },
        {
            "title": "The Shawshank Redemption",
            "year": 1994,
            "duration": 142,
            "genres": ["Drama"],
            "countries": ["USA"],
            "description": "Two imprisoned men bond over a number of years.",
            "rating": 4.8,
            "watched_at": "2023-03-10",
        },
    ]


@pytest.fixture
def sample_recommendation_request():
    """Sample recommendation request for testing."""
    from schemas.recommendation import RecommendationRequest

    return RecommendationRequest(
        user_id="test_user",
        filters={
            "genres": ["Action", "Sci-Fi"],
            "years": [2010, 2015],
            "countries": ["USA"],
        },
    )

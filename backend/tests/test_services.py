import pytest

from domain.entities import Cluster, Embedding, Movie, UserProfile
from services.clustering_service import SklearnClusteringService
from services.filtering_service import FilteringService
from services.llm_service import OpenAILLMService
from services.recommendation_service import RecommendationService


class TestRecommendationService:
    """Test suite for RecommendationService."""

    @pytest.fixture
    def recommendation_service(self, mock_logger):
        return RecommendationService(logger=mock_logger)

    @pytest.fixture
    def sample_user_profile(self):
        """Create a sample user profile with clusters."""
        return UserProfile(
            user_id="test_user",
            movies=[],
            clusters=[
                Cluster(
                    centroid=Embedding([0.1, 0.2, 0.3, 0.4, 0.5] * 100),
                    movies=[],
                    average_rating=4.5,
                    count=1,
                ),
                Cluster(
                    centroid=Embedding([0.6, 0.7, 0.8, 0.9, 1.0] * 100),
                    movies=[],
                    average_rating=4.8,
                    count=1,
                ),
            ],
        )

    @pytest.fixture
    def sample_candidates(self):
        """Create sample movie candidates."""
        return [
            Movie(
                id="1",
                title="Interstellar",
                year=2014,
                duration=169,
                genres=["Adventure", "Drama", "Sci-Fi"],
                countries=["USA", "UK"],
                description="A team of explorers travel through a wormhole in space.",
                embedding=Embedding([0.15, 0.25, 0.35, 0.45, 0.55] * 100),
            ),
            Movie(
                id="2",
                title="The Dark Knight",
                year=2008,
                duration=152,
                genres=["Action", "Crime", "Drama"],
                countries=["USA", "UK"],
                description="When the menace known as the Joker wreaks havoc on Gotham City.",
                embedding=Embedding([0.65, 0.75, 0.85, 0.95, 1.05] * 100),
            ),
        ]

    def test_recommend_success(
        self, recommendation_service, sample_user_profile, sample_candidates
    ):
        # Act
        result = recommendation_service.recommend(
            sample_user_profile, sample_candidates
        )

        # Assert
        assert len(result) == 2
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], float) for item in result)
        assert all(isinstance(item[1], Movie) for item in result)
        assert all(0 <= item[0] <= 1 for item in result)

    def test_recommend_with_movies_without_embeddings(
        self, recommendation_service, sample_user_profile, sample_candidates
    ):
        # Arrange
        sample_candidates[0].embedding = None

        # Act
        result = recommendation_service.recommend(
            sample_user_profile, sample_candidates
        )

        # Assert
        assert len(result) == 1
        assert result[0][1].title == "The Dark Knight"

    def test_recommend_with_empty_candidates(
        self, recommendation_service, sample_user_profile
    ):
        # Act
        result = recommendation_service.recommend(sample_user_profile, [])

        # Assert
        assert len(result) == 0

    def test_recommend_with_empty_user_profile(
        self, recommendation_service, sample_candidates
    ):
        # Arrange
        empty_profile = UserProfile(user_id="test_user", movies=[], clusters=[])

        # Act & Assert
        with pytest.raises(ValueError):
            recommendation_service.recommend(empty_profile, sample_candidates)

    def test_recommend_similarity_calculation(
        self, recommendation_service, sample_user_profile, sample_candidates
    ):
        # Act
        result = recommendation_service.recommend(
            sample_user_profile, sample_candidates
        )

        # Assert
        assert result[0][0] > 0
        assert result[1][0] > 0


class TestClusteringService:
    """Test suite for SklearnClusteringService."""

    @pytest.fixture
    def clustering_service(self, mock_logger):
        return SklearnClusteringService(logger=mock_logger)

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return [
            Embedding([0.1, 0.2, 0.3, 0.4, 0.5] * 100),
            Embedding([0.2, 0.3, 0.4, 0.5, 0.6] * 100),
            Embedding([0.3, 0.4, 0.5, 0.6, 0.7] * 100),
            Embedding([0.4, 0.5, 0.6, 0.7, 0.8] * 100),
            Embedding([0.5, 0.6, 0.7, 0.8, 0.9] * 100),
        ]

    @pytest.fixture
    def sample_ratings(self):
        return [4.5, 4.7, 4.3, 4.8, 4.6]

    @pytest.fixture
    def sample_dates(self):
        return ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-12"]

    @pytest.fixture
    def sample_movies(self):
        """Create sample movies for testing."""
        return [
            Movie(
                id="1",
                title="The Matrix",
                year=1999,
                duration=136,
                genres=["Action", "Sci-Fi"],
                countries=["USA"],
                description="A computer hacker learns about reality.",
                embedding=Embedding([0.1, 0.2, 0.3, 0.4, 0.5] * 100),
            ),
            Movie(
                id="2",
                title="Inception",
                year=2010,
                duration=148,
                genres=["Action", "Sci-Fi", "Thriller"],
                countries=["USA", "UK"],
                description="A thief who steals corporate secrets.",
                embedding=Embedding([0.2, 0.3, 0.4, 0.5, 0.6] * 100),
            ),
            Movie(
                id="3",
                title="The Shawshank Redemption",
                year=1994,
                duration=142,
                genres=["Drama"],
                countries=["USA"],
                description="Two imprisoned men bond over a number of years.",
                embedding=Embedding([0.3, 0.4, 0.5, 0.6, 0.7] * 100),
            ),
            Movie(
                id="4",
                title="Pulp Fiction",
                year=1994,
                duration=154,
                genres=["Crime", "Drama"],
                countries=["USA"],
                description="The lives of two mob hitmen, a boxer, a gangster and his wife intertwine.",
                embedding=Embedding([0.4, 0.5, 0.6, 0.7, 0.8] * 100),
            ),
            Movie(
                id="5",
                title="Forrest Gump",
                year=1994,
                duration=142,
                genres=["Drama", "Romance"],
                countries=["USA"],
                description="The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal.",
                embedding=Embedding([0.5, 0.6, 0.7, 0.8, 0.9] * 100),
            ),
        ]

    def test_cluster_small_dataset(
        self,
        clustering_service,
        sample_embeddings,
        sample_ratings,
        sample_dates,
        sample_movies,
    ):
        # Act
        result = clustering_service.cluster(
            sample_embeddings, sample_ratings, sample_dates, sample_movies
        )

        # Assert
        assert len(result) > 0
        assert all(isinstance(cluster, Cluster) for cluster in result)

    def test_cluster_medium_dataset(
        self,
        clustering_service,
        sample_embeddings,
        sample_ratings,
        sample_dates,
        sample_movies,
    ):
        # Arrange
        medium_embeddings = sample_embeddings * 20
        medium_ratings = sample_ratings * 20
        medium_dates = sample_dates * 20
        medium_movies = sample_movies * 20

        # Act
        result = clustering_service.cluster(
            medium_embeddings, medium_ratings, medium_dates, medium_movies
        )

        # Assert
        assert len(result) > 0
        assert all(isinstance(cluster, Cluster) for cluster in result)

    def test_cluster_large_dataset(
        self,
        clustering_service,
        sample_embeddings,
        sample_ratings,
        sample_dates,
        sample_movies,
    ):
        # Arrange
        large_embeddings = sample_embeddings * 120
        large_ratings = sample_ratings * 120
        large_dates = sample_dates * 120
        large_movies = sample_movies * 120

        # Act
        result = clustering_service.cluster(
            large_embeddings, large_ratings, large_dates, large_movies
        )

        # Assert
        assert len(result) > 0
        assert all(isinstance(cluster, Cluster) for cluster in result)

    def test_cluster_with_empty_data(self, clustering_service):
        # Act & Assert
        with pytest.raises(ZeroDivisionError):
            clustering_service.cluster([], [], [], [])

    def test_cluster_with_single_movie(
        self,
        clustering_service,
        sample_embeddings,
        sample_ratings,
        sample_dates,
        sample_movies,
    ):
        # Arrange
        single_embedding = [sample_embeddings[0]]
        single_rating = [sample_ratings[0]]
        single_date = [sample_dates[0]]
        single_movie = [sample_movies[0]]

        # Act
        result = clustering_service.cluster(
            single_embedding, single_rating, single_date, single_movie
        )

        # Assert
        assert len(result) == 1
        assert result[0].count == 1


class TestFilteringService:
    """Test suite for FilteringService."""

    @pytest.fixture
    def filtering_service(self, mock_logger):
        return FilteringService(logger=mock_logger)

    @pytest.fixture
    def sample_movies(self):
        """Create sample movies for testing."""
        return [
            Movie(
                id="1",
                title="The Matrix",
                year=1999,
                duration=136,
                genres=["Action", "Sci-Fi"],
                countries=["USA"],
                description="A computer hacker learns about reality.",
                embedding=None,
            ),
            Movie(
                id="2",
                title="Inception",
                year=2010,
                duration=148,
                genres=["Action", "Sci-Fi", "Thriller"],
                countries=["USA", "UK"],
                description="A thief who steals corporate secrets.",
                embedding=None,
            ),
            Movie(
                id="3",
                title="The Shawshank Redemption",
                year=1994,
                duration=142,
                genres=["Drama"],
                countries=["USA"],
                description="Two imprisoned men bond over a number of years.",
                embedding=None,
            ),
        ]

    @pytest.fixture
    def watched_movies(self):
        """Create sample watched movies."""
        return [
            Movie(
                id="watched1",
                title="The Matrix",
                year=1999,
                duration=136,
                genres=["Action", "Sci-Fi"],
                countries=["USA"],
                description="A computer hacker learns about reality.",
                embedding=None,
            ),
        ]

    def test_filter_with_watched_movies(
        self, filtering_service, sample_movies, watched_movies
    ):
        # Arrange
        filters = {"watched_movies": watched_movies}

        # Act
        result = filtering_service.filter(sample_movies, filters)

        # Assert
        assert len(result) == 2
        assert all(movie.title != "The Matrix" for movie in result)

    def test_filter_with_genre_filter(self, filtering_service, sample_movies):
        # Arrange
        filters = {"genres": ["Action"]}

        # Act
        result = filtering_service.filter(sample_movies, filters)

        # Assert
        assert len(result) == 2
        assert all("Action" in movie.genres for movie in result)

    def test_filter_with_year_filter(self, filtering_service, sample_movies):
        # Arrange
        filters = {"years": [1999, 2010]}

        # Act
        result = filtering_service.filter(sample_movies, filters)

        # Assert
        assert len(result) == 2
        assert all(movie.year in [1999, 2010] for movie in result)

    def test_filter_with_country_filter(self, filtering_service, sample_movies):
        # Arrange
        filters = {"countries": ["UK"]}

        # Act
        result = filtering_service.filter(sample_movies, filters)

        # Assert
        assert len(result) == 1
        assert "UK" in result[0].countries

    def test_filter_with_duration_filter(self, filtering_service, sample_movies):
        # Arrange
        filters = {"durations": [140]}

        # Act
        result = filtering_service.filter(sample_movies, filters)

        # Assert
        assert len(result) == 1
        assert all(movie.duration <= 140 for movie in result)

    def test_filter_with_multiple_filters(self, filtering_service, sample_movies):
        # Arrange
        filters = {"genres": ["Action"], "years": [1999], "countries": ["USA"]}

        # Act
        result = filtering_service.filter(sample_movies, filters)

        # Assert
        assert len(result) == 1
        assert result[0].title == "The Matrix"

    def test_filter_with_no_filters(self, filtering_service, sample_movies):
        # Arrange
        filters = {}

        # Act
        result = filtering_service.filter(sample_movies, filters)

        # Assert
        assert len(result) == 3

    def test_filter_with_empty_movies(self, filtering_service):
        # Act
        result = filtering_service.filter([], {})

        # Assert
        assert len(result) == 0

    def test_filter_deduplication(self, filtering_service, sample_movies):
        # Arrange
        duplicate_movies = sample_movies + sample_movies

        # Act
        result = filtering_service.filter(duplicate_movies, {})

        # Assert
        assert len(result) == 3
        titles = [movie.title for movie in result]
        assert len(titles) == len(set(titles))


class TestLLMService:
    """Test suite for OpenAILLMService."""

    @pytest.fixture
    def llm_service(self, mock_logger):
        return OpenAILLMService(logger=mock_logger)

    @pytest.fixture
    def sample_user_profile(self):
        """Create a sample user profile for testing."""
        return UserProfile(
            user_id="test_user",
            movies=[],
            clusters=[
                Cluster(
                    centroid=Embedding([0.1, 0.2, 0.3, 0.4, 0.5] * 100),
                    movies=[
                        Movie(
                            id="1",
                            title="The Matrix",
                            year=1999,
                            duration=136,
                            genres=["Action", "Sci-Fi"],
                            countries=["USA"],
                            description="A computer hacker learns about reality.",
                            embedding=None,
                        ),
                    ],
                    average_rating=4.5,
                    count=1,
                ),
            ],
        )

    @pytest.fixture
    def sample_candidates(self):
        """Create sample movie candidates for testing."""
        return [
            Movie(
                id="1",
                title="Interstellar",
                year=2014,
                duration=169,
                genres=["Adventure", "Drama", "Sci-Fi"],
                countries=["USA", "UK"],
                description="A team of explorers travel through a wormhole in space.",
                embedding=None,
            ),
            Movie(
                id="2",
                title="The Dark Knight",
                year=2008,
                duration=152,
                genres=["Action", "Crime", "Drama"],
                countries=["USA", "UK"],
                description="When the menace known as the Joker wreaks havoc on Gotham City.",
                embedding=None,
            ),
        ]

    @pytest.mark.skip(reason="LLM integration test - requires complex mocking")
    def test_rerank_success(self, llm_service, sample_user_profile, sample_candidates):
        # Act
        result = llm_service.rerank(sample_user_profile, sample_candidates)

        # Assert
        assert len(result) == 2
        assert all(isinstance(item, dict) for item in result)
        assert all("title" in item for item in result)
        assert all("year" in item for item in result)
        assert all("genres" in item for item in result)
        assert all("justification" in item for item in result)

    @pytest.mark.skip(reason="LLM integration test - requires complex mocking")
    def test_rerank_with_empty_candidates(self, llm_service, sample_user_profile):
        # Act
        result = llm_service.rerank(sample_user_profile, [])

        # Assert
        assert len(result) == 10

    def test_rerank_with_empty_user_profile(self, llm_service, sample_candidates):
        # Arrange
        empty_profile = UserProfile(user_id="test_user", movies=[], clusters=[])

        # Act
        result = llm_service.rerank(empty_profile, sample_candidates)

        # Assert
        assert len(result) == 2

    def test_build_taste_summary(self, llm_service, sample_user_profile):
        # Act
        result = llm_service._build_taste_summary(sample_user_profile)

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Taste group" in result
        assert "Action" in result
        assert "Sci-Fi" in result
        assert "USA" in result

    def test_build_taste_summary_with_empty_profile(self, llm_service):
        # Arrange
        empty_profile = UserProfile(user_id="test_user", movies=[], clusters=[])

        # Act
        result = llm_service._build_taste_summary(empty_profile)

        # Assert
        assert isinstance(result, str)
        assert len(result) == 0

    def test_build_taste_summary_with_multiple_clusters(self, llm_service):
        # Arrange
        multi_cluster_profile = UserProfile(
            user_id="test_user",
            movies=[],
            clusters=[
                Cluster(
                    centroid=Embedding([0.1, 0.2, 0.3, 0.4, 0.5] * 100),
                    movies=[
                        Movie(
                            id="1",
                            title="The Matrix",
                            year=1999,
                            duration=136,
                            genres=["Action", "Sci-Fi"],
                            countries=["USA"],
                            description="A computer hacker learns about reality.",
                            embedding=None,
                        ),
                    ],
                    average_rating=4.5,
                    count=1,
                ),
                Cluster(
                    centroid=Embedding([0.6, 0.7, 0.8, 0.9, 1.0] * 100),
                    movies=[
                        Movie(
                            id="2",
                            title="The Shawshank Redemption",
                            year=1994,
                            duration=142,
                            genres=["Drama"],
                            countries=["USA"],
                            description="Two imprisoned men bond over a number of years.",
                            embedding=None,
                        ),
                    ],
                    average_rating=4.8,
                    count=1,
                ),
            ],
        )

        # Act
        result = llm_service._build_taste_summary(multi_cluster_profile)

        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        assert result.count("Taste group") == 2
        assert "Action" in result
        assert "Drama" in result

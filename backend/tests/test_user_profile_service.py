from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.entities import MovieHistoryItem, UserProfile
from services.user_profile_service import UserProfileService


class TestUserProfileService:
    """Test suite for UserProfileService covering profile creation and retrieval."""

    @pytest.mark.asyncio
    async def test_get_profile_success(
        self, mock_user_profile_service: UserProfileService
    ):
        """Test successful profile retrieval."""
        # Act
        result = await mock_user_profile_service.get_profile("test_user")

        # Assert
        assert isinstance(result, UserProfile)
        assert result.user_id == "test_user"
        assert len(result.movies) == 2
        assert len(result.clusters) == 1

    @pytest.mark.asyncio
    async def test_get_profile_not_found(
        self, mock_user_profile_service: UserProfileService
    ):
        """Test profile retrieval when profile doesn't exist."""
        # Arrange
        mock_user_profile_service.vectorstore.get_user_profile = AsyncMock(
            return_value=None
        )

        # Act
        result = await mock_user_profile_service.get_profile("nonexistent_user")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_create_and_save_profile_success(
        self, mock_user_profile_service: UserProfileService, sample_movies_data
    ):
        """Test successful profile creation and saving."""
        # Act
        result = await mock_user_profile_service.create_and_save_profile(
            "test_user", sample_movies_data
        )

        # Assert
        assert isinstance(result, UserProfile)
        assert result.user_id == "test_user"
        assert len(result.movies) == 2
        assert len(result.clusters) == 1

        mock_user_profile_service.field_detector.convert_movies_batch.assert_called_once_with(
            sample_movies_data
        )
        mock_user_profile_service.embedder.embed.assert_called_once()
        mock_user_profile_service.clusterer.cluster.assert_called_once()
        mock_user_profile_service.vectorstore.save_user_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_and_save_profile_with_empty_data(
        self, mock_user_profile_service: UserProfileService
    ):
        """Test profile creation with empty movies data."""
        # Arrange
        empty_data = []

        # Act
        result = await mock_user_profile_service.create_and_save_profile(
            "test_user", empty_data
        )

        # Assert
        assert isinstance(result, UserProfile)
        assert result.user_id == "test_user"
        assert len(result.movies) == 2

    @pytest.mark.asyncio
    async def test_create_and_save_profile_with_field_detection_failure(
        self, mock_user_profile_service: UserProfileService, sample_movies_data
    ):
        """Test profile creation when field detection fails."""
        # Arrange
        mock_user_profile_service.field_detector.convert_movies_batch = AsyncMock(
            side_effect=Exception("Field detection failed")
        )

        # Act & Assert
        with pytest.raises(Exception, match="Field detection failed"):
            await mock_user_profile_service.create_and_save_profile(
                "test_user", sample_movies_data
            )

    @pytest.mark.asyncio
    async def test_create_and_save_profile_with_embedding_failure(
        self, mock_user_profile_service: UserProfileService, sample_movies_data
    ):
        """Test profile creation when embedding generation fails."""
        # Arrange
        mock_user_profile_service.embedder.embed = AsyncMock(
            side_effect=Exception("Embedding failed")
        )

        # Act & Assert
        with pytest.raises(Exception, match="Embedding failed"):
            await mock_user_profile_service.create_and_save_profile(
                "test_user", sample_movies_data
            )

    @pytest.mark.asyncio
    async def test_create_and_save_profile_with_clustering_failure(
        self, mock_user_profile_service: UserProfileService, sample_movies_data
    ):
        """Test profile creation when clustering fails."""
        # Arrange
        mock_user_profile_service.clusterer.cluster = MagicMock(
            side_effect=Exception("Clustering failed")
        )

        # Act & Assert
        with pytest.raises(Exception, match="Clustering failed"):
            await mock_user_profile_service.create_and_save_profile(
                "test_user", sample_movies_data
            )

    @pytest.mark.asyncio
    async def test_create_and_save_profile_with_save_failure(
        self, mock_user_profile_service: UserProfileService, sample_movies_data
    ):
        """Test profile creation when saving to database fails."""
        # Arrange
        mock_user_profile_service.vectorstore.save_user_profile = AsyncMock(
            side_effect=Exception("Save failed")
        )

        # Act & Assert
        with pytest.raises(Exception, match="Save failed"):
            await mock_user_profile_service.create_and_save_profile(
                "test_user", sample_movies_data
            )

    @pytest.mark.asyncio
    async def test_load_watch_history_from_content_success(
        self, mock_user_profile_service: UserProfileService, sample_movies_data
    ):
        """Test loading watch history from content."""
        # Act
        result = await mock_user_profile_service._load_watch_history_from_content(
            sample_movies_data
        )

        # Assert
        assert len(result) == 2
        assert all(isinstance(item, MovieHistoryItem) for item in result)
        assert result[0].title == "The Matrix"
        assert result[1].title == "Inception"

        mock_user_profile_service.field_detector.convert_movies_batch.assert_called_once_with(
            sample_movies_data
        )

    @pytest.mark.asyncio
    async def test_load_watch_history_with_excluded_movies(
        self, mock_user_profile_service: UserProfileService, sample_movies_data
    ):
        """Test loading watch history with some movies excluded."""
        # Arrange
        converted_movies = [
            {
                "title": "The Matrix",
                "year": 1999,
                "duration": 136,
                "genres": ["Action", "Sci-Fi"],
                "countries": ["USA"],
                "description": "A computer hacker learns about reality.",
                "rating": 4.5,
                "watched_at": "2023-01-15",
            }
        ]
        excluded_movies = [
            {
                "original": {"title": "Invalid Movie"},
                "enriched": {},
                "missing_fields": ["year", "genres"],
            }
        ]
        mock_user_profile_service.field_detector.convert_movies_batch.return_value = (
            converted_movies,
            excluded_movies,
        )

        # Act
        result = await mock_user_profile_service._load_watch_history_from_content(
            sample_movies_data
        )

        # Assert
        assert len(result) == 1
        assert result[0].title == "The Matrix"

    @pytest.mark.asyncio
    async def test_load_watch_history_with_validation_failure(
        self, mock_user_profile_service: UserProfileService, sample_movies_data
    ):
        """Test loading watch history when validation fails."""
        # Arrange
        converted_movies = [
            {
                "title": "Invalid Movie",
                "year": 1999,
                "duration": 136,
                "genres": ["Action", "Sci-Fi"],
                "countries": ["USA"],
                "description": "A computer hacker learns about reality.",
                "rating": 4.5,
                "watched_at": "2023-01-15",
            }
        ]
        mock_user_profile_service.field_detector.convert_movies_batch.return_value = (
            converted_movies,
            [],
        )
        mock_user_profile_service.field_detector.validate_movie_data.return_value = (
            False
        )

        # Act
        result = await mock_user_profile_service._load_watch_history_from_content(
            sample_movies_data
        )

        # Assert
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_build_user_profile_from_history_success(
        self, mock_user_profile_service: UserProfileService
    ):
        """Test building user profile from watch history."""
        # Arrange
        watch_history = [
            MovieHistoryItem(
                title="The Matrix",
                rating=4.5,
                year=1999,
                duration=136,
                genres=["Action", "Sci-Fi"],
                countries=["USA"],
                description="A computer hacker learns about reality.",
                watched_at="2023-01-15",
            ),
            MovieHistoryItem(
                title="Inception",
                rating=4.7,
                year=2010,
                duration=148,
                genres=["Action", "Sci-Fi", "Thriller"],
                countries=["USA", "UK"],
                description="A thief who steals corporate secrets.",
                watched_at="2023-02-20",
            ),
        ]

        # Act
        result = await mock_user_profile_service._build_user_profile_from_history(
            "test_user", watch_history
        )

        # Assert
        assert isinstance(result, UserProfile)
        assert result.user_id == "test_user"
        assert len(result.movies) == 2
        assert len(result.clusters) == 1

        mock_user_profile_service.embedder.embed.assert_called_once()
        mock_user_profile_service.clusterer.cluster.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_user_profile_with_high_rated_filtering(
        self, mock_user_profile_service: UserProfileService
    ):
        """Test building user profile with high-rated movie filtering."""
        # Arrange
        watch_history = [
            MovieHistoryItem(
                title="The Matrix",
                rating=4.5,
                year=1999,
                duration=136,
                genres=["Action", "Sci-Fi"],
                countries=["USA"],
                description="A computer hacker learns about reality.",
                watched_at="2023-01-15",
            ),
            MovieHistoryItem(
                title="Bad Movie",
                rating=2.0,
                year=2020,
                duration=90,
                genres=["Comedy"],
                countries=["USA"],
                description="A really bad movie.",
                watched_at="2023-03-15",
            ),
        ]

        # Act
        result = await mock_user_profile_service._build_user_profile_from_history(
            "test_user", watch_history
        )

        # Assert
        assert isinstance(result, UserProfile)
        assert result.user_id == "test_user"
        assert len(result.movies) == 1
        assert result.movies[0].title == "The Matrix"

    @pytest.mark.asyncio
    async def test_build_user_profile_with_no_high_rated_movies(
        self, mock_user_profile_service: UserProfileService
    ):
        """Test building user profile when no movies meet the high-rated threshold."""
        # Arrange
        watch_history = [
            MovieHistoryItem(
                title="Bad Movie 1",
                rating=2.0,
                year=2020,
                duration=90,
                genres=["Comedy"],
                countries=["USA"],
                description="A really bad movie.",
                watched_at="2023-01-15",
            ),
            MovieHistoryItem(
                title="Bad Movie 2",
                rating=2.5,
                year=2021,
                duration=95,
                genres=["Drama"],
                countries=["USA"],
                description="Another bad movie.",
                watched_at="2023-02-20",
            ),
        ]

        # Act
        result = await mock_user_profile_service._build_user_profile_from_history(
            "test_user", watch_history
        )

        # Assert
        assert isinstance(result, UserProfile)
        assert result.user_id == "test_user"
        assert len(result.movies) == 2
        mock_user_profile_service.embedder.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_user_profile_logging_verification(
        self, mock_user_profile_service: UserProfileService
    ):
        """Test that logging is properly configured and working."""
        # Arrange
        watch_history = [
            MovieHistoryItem(
                title="The Matrix",
                rating=4.5,
                year=1999,
                duration=136,
                genres=["Action", "Sci-Fi"],
                countries=["USA"],
                description="A computer hacker learns about reality.",
                watched_at="2023-01-15",
            ),
        ]

        # Act
        result = await mock_user_profile_service._build_user_profile_from_history(
            "test_user", watch_history
        )

        # Assert
        assert isinstance(result, UserProfile)
        assert hasattr(mock_user_profile_service, "logger")
        assert mock_user_profile_service.logger is not None

    @pytest.mark.asyncio
    async def test_create_and_save_profile_logging_verification(
        self, mock_user_profile_service: UserProfileService, sample_movies_data
    ):
        """Test that logging is properly configured and working during profile creation."""
        # Act
        result = await mock_user_profile_service.create_and_save_profile(
            "test_user", sample_movies_data
        )

        # Assert
        assert isinstance(result, UserProfile)
        assert hasattr(mock_user_profile_service, "logger")
        assert mock_user_profile_service.logger is not None

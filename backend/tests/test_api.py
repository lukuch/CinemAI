import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.main import app
from core.service_factories import get_user_profile_service
from services.user_profile_service import UserProfileService


@pytest.fixture(autouse=True)
def override_user_profile_service(monkeypatch):
    mock_profile_service = AsyncMock()
    mock_profile_service.get_profile.return_value = MagicMock(user_id="test_user", movies=[], clusters=[])

    def _get_user_profile_service():
        return mock_profile_service

    app.dependency_overrides[get_user_profile_service] = _get_user_profile_service
    yield
    app.dependency_overrides = {}


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_movies_data(self):
        """Sample movies data for testing."""
        return [
            {
                "title": "The Matrix",
                "year": 1999,
                "duration": 136,
                "genres": ["Action", "Sci-Fi"],
                "countries": ["USA"],
                "description": "A computer hacker learns about reality.",
                "rating": 4.5,
                "watched_at": "2023-01-15",
            },
            {
                "title": "Inception",
                "year": 2010,
                "duration": 148,
                "genres": ["Action", "Sci-Fi", "Thriller"],
                "countries": ["USA", "UK"],
                "description": "A thief who steals corporate secrets.",
                "rating": 4.7,
                "watched_at": "2023-02-20",
            },
        ]

    @patch("api.routes.get_user_profile_service")
    @patch("api.routes.get_recommendation_manager")
    def test_upload_watch_history_success(
        self, mock_recommendation_manager, mock_user_profile_service, client, sample_movies_data
    ):
        # Arrange
        mock_user_profile_service.return_value = AsyncMock()
        mock_recommendation_manager.return_value = AsyncMock()

        json_content = """[
            {
                "title": "The Matrix",
                "year": 1999,
                "duration": 136,
                "genres": ["Action", "Sci-Fi"],
                "countries": ["USA"],
                "description": "A computer hacker learns about reality.",
                "rating": 4.5,
                "watched_at": "2023-01-15"
            },
            {
                "title": "Inception",
                "year": 2010,
                "duration": 148,
                "genres": ["Action", "Sci-Fi", "Thriller"],
                "countries": ["USA", "UK"],
                "description": "A thief who steals corporate secrets.",
                "rating": 4.7,
                "watched_at": "2023-02-20"
            }
        ]"""

        files = {"file": ("watch_history.json", io.BytesIO(json_content.encode("utf-8")), "application/json")}

        # Act
        response = client.post("/upload-watch-history?user_id=test_user", files=files)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "movies_count" in data

    @patch("api.routes.get_user_profile_service")
    @patch("api.routes.get_recommendation_manager")
    def test_upload_watch_history_without_movies(self, mock_recommendation_manager, mock_user_profile_service, client):
        # Arrange
        mock_user_profile_service.return_value = AsyncMock()
        mock_recommendation_manager.return_value = AsyncMock()

        json_content = "[]"
        files = {"file": ("watch_history.json", io.BytesIO(json_content.encode("utf-8")), "application/json")}

        # Act
        response = client.post("/upload-watch-history?user_id=test_user", files=files)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    @patch.object(UserProfileService, "get_profile", new_callable=AsyncMock)
    @patch("api.routes.get_user_profile_service")
    @patch("api.routes.get_recommendation_manager")
    def test_get_recommendations_with_empty_filters(
        self, mock_recommendation_manager, mock_user_profile_service, mock_get_profile, client
    ):
        # Arrange
        movie = SimpleNamespace(
            title="Movie 1",
            rating=5.0,
            year=2020,
            duration=120,
            genres=["Action"],
            countries=["USA"],
            description="desc",
            watched_at="2023-01-01",
        )
        centroid = SimpleNamespace(vector=np.zeros(3072))
        cluster = SimpleNamespace(centroid=centroid, movies=[movie], count=1, average_rating=5.0)
        mock_get_profile.return_value = SimpleNamespace(user_id="test_user", movies=[movie], clusters=[cluster])
        mock_user_profile_service.return_value = AsyncMock()
        mock_recommendation_manager.return_value = AsyncMock()
        mock_recommendation_manager.return_value.recommend.return_value = MagicMock(
            recommendations=[
                {
                    "title": "Interstellar",
                    "year": 2014,
                    "genres": ["Adventure", "Drama", "Sci-Fi"],
                    "countries": ["USA", "UK"],
                    "description": "A team of explorers travel through a wormhole in space.",
                    "duration": 169,
                    "similarity": 0.85,
                    "justification": "Based on your love for sci-fi films.",
                }
            ]
        )

        # Act
        response = client.post("/recommend", json={"user_id": "test_user", "filters": {}})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0

    @patch.object(UserProfileService, "get_profile", new_callable=AsyncMock)
    @patch("api.routes.get_user_profile_service")
    @patch("api.routes.get_recommendation_manager")
    def test_get_recommendations_with_nonexistent_user(
        self, mock_recommendation_manager, mock_user_profile_service, mock_get_profile, client
    ):
        # Arrange
        mock_get_profile.return_value = None
        mock_user_profile_service.return_value = AsyncMock()
        mock_recommendation_manager.return_value = AsyncMock()

        # Act
        response = client.post("/recommend", json={"user_id": "nonexistent_user", "filters": {}})

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @patch("api.routes.get_user_profile_service")
    @patch("api.routes.get_recommendation_manager")
    def test_get_recommendations_with_invalid_filters(self, mock_recommendation_manager, mock_user_profile_service, client):
        # Arrange
        mock_user_profile_service.return_value = AsyncMock()
        mock_recommendation_manager.return_value = AsyncMock()

        # Act
        response = client.post("/recommend", json={"user_id": "test_user", "min_year": "invalid"})

        # Assert
        assert response.status_code == 422

    @patch.object(UserProfileService, "get_profile", new_callable=AsyncMock)
    @patch("api.routes.get_user_profile_service")
    @patch("api.routes.get_recommendation_manager")
    def test_get_recommendations_with_complex_filters(
        self, mock_recommendation_manager, mock_user_profile_service, mock_get_profile, client
    ):
        # Arrange
        movie = SimpleNamespace(
            title="Movie 1",
            rating=5.0,
            year=2020,
            duration=120,
            genres=["Action"],
            countries=["USA"],
            description="desc",
            watched_at="2023-01-01",
        )
        centroid = SimpleNamespace(vector=np.zeros(3072))
        cluster = SimpleNamespace(centroid=centroid, movies=[movie], count=1, average_rating=5.0)
        mock_get_profile.return_value = SimpleNamespace(user_id="test_user", movies=[movie], clusters=[cluster])
        mock_user_profile_service.return_value = AsyncMock()
        mock_recommendation_manager.return_value = AsyncMock()
        mock_recommendation_manager.return_value.recommend.return_value = MagicMock(
            recommendations=[
                {
                    "title": "The Dark Knight",
                    "year": 2008,
                    "genres": ["Action", "Crime", "Drama"],
                    "countries": ["USA", "UK"],
                    "description": "When the menace known as the Joker wreaks havoc on Gotham City.",
                    "duration": 152,
                    "similarity": 0.78,
                    "justification": "Given your appreciation for complex narratives.",
                }
            ]
        )

        # Act
        response = client.post(
            "/recommend",
            json={
                "user_id": "test_user",
                "filters": {"genres": ["Action"], "min_year": 2000, "max_year": 2010, "countries": ["USA"]},
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data

    @patch.object(UserProfileService, "create_and_save_profile", new_callable=AsyncMock)
    @patch("api.routes.get_user_profile_service")
    @patch("api.routes.get_recommendation_manager")
    def test_upload_watch_history_with_large_dataset(
        self, mock_recommendation_manager, mock_user_profile_service, mock_create_and_save_profile, client
    ):
        # Arrange
        mock_user_profile_service.return_value = AsyncMock()
        mock_recommendation_manager.return_value = AsyncMock()
        mock_create_and_save_profile.return_value = None

        json_content = "["
        for i in range(100):
            rating = 1.0 + (i % 5)
            day = (i % 28) + 1
            json_content += f'{{"title": "Movie {i}","year": 2000,"duration": 120,"genres": ["Action"],"countries": ["USA"],"description": "A test movie {i}.","rating": {rating},"watched_at": "2023-01-{day:02d}"}}'
            if i < 99:
                json_content += ","
        json_content += "]"

        files = {"file": ("watch_history.json", io.BytesIO(json_content.encode("utf-8")), "application/json")}

        # Act
        response = client.post("/upload-watch-history?user_id=test_user", files=files)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    @patch.object(UserProfileService, "get_profile", new_callable=AsyncMock)
    @patch("api.routes.get_user_profile_service")
    @patch("api.routes.get_recommendation_manager")
    def test_upload_watch_history_validation(
        self, mock_recommendation_manager, mock_user_profile_service, mock_get_profile, client
    ):
        # Arrange
        mock_user_profile_service.return_value = AsyncMock()
        mock_recommendation_manager.return_value = AsyncMock()

        json_content = "invalid json content"
        files = {"file": ("watch_history.json", io.BytesIO(json_content.encode("utf-8")), "application/json")}

        # Act
        response = client.post("/upload-watch-history?user_id=test_user", files=files)

        # Assert
        assert response.status_code == 400

    @patch.object(UserProfileService, "get_profile", new_callable=AsyncMock)
    @patch("api.routes.get_user_profile_service")
    @patch("api.routes.get_recommendation_manager")
    def test_get_profile(self, mock_recommendation_manager, mock_user_profile_service, mock_get_profile, client):
        # Arrange
        from domain.entities import UserProfile
        from schemas.watch_history import MovieHistoryItem

        profile = UserProfile(user_id="test_user", movies=[], clusters=[])
        mock_get_profile.return_value = profile
        mock_user_profile_service.return_value = AsyncMock()
        mock_recommendation_manager.return_value = AsyncMock()

        # Act
        response = client.get("/profiles/test_user")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert data["user_id"] == "test_user"

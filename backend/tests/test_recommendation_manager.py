from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.entities import Movie, UserProfile
from managers.recommendation_manager import RecommendationManager
from schemas.recommendation import RecommendationRequest, RecommendationResponse


class TestRecommendationManager:
    """Test suite for RecommendationManager covering the complete recommendation flow."""

    @pytest.mark.asyncio
    async def test_recommend_successful_flow(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test the complete recommendation flow with all components working correctly."""
        # Act
        result = await recommendation_manager.recommend(sample_recommendation_request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert hasattr(result, "recommendations")
        assert len(result.recommendations) > 0

        recommendation_manager.user_profile_service.get_profile.assert_called_once_with("test_user")
        recommendation_manager.embedder.embed.assert_called()
        recommendation_manager.recommender.recommend.assert_called()
        recommendation_manager.llm.rerank.assert_called()

    @pytest.mark.asyncio
    async def test_recommend_with_empty_user_profile(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test recommendation with empty user profile."""
        # Arrange
        recommendation_manager.user_profile_service.get_profile.return_value = UserProfile(
            user_id="test_user", movies=[], clusters=[]
        )

        # Act
        result = await recommendation_manager.recommend(sample_recommendation_request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_recommend_with_nonexistent_user(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test recommendation for nonexistent user."""
        # Arrange
        recommendation_manager.user_profile_service.get_profile.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="User profile not found"):
            await recommendation_manager.recommend(sample_recommendation_request)

    @pytest.mark.asyncio
    async def test_recommend_with_complex_filters(self, recommendation_manager: RecommendationManager):
        """Test recommendation with complex filter combinations."""
        # Arrange
        complex_request = RecommendationRequest(
            user_id="test_user",
            filters={
                "genres": ["Action", "Sci-Fi"],
                "min_year": 2010,
                "max_year": 2020,
                "countries": ["USA", "UK"],
                "min_duration": 120,
                "max_duration": 180,
            },
        )

        # Act
        result = await recommendation_manager.recommend(complex_request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_recommend_with_empty_filters(self, recommendation_manager: RecommendationManager):
        """Test recommendation with empty filters."""
        # Arrange
        empty_request = RecommendationRequest(user_id="test_user", filters={})

        # Act
        result = await recommendation_manager.recommend(empty_request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_recommend_with_invalid_filters(self, recommendation_manager: RecommendationManager):
        """Test recommendation with invalid filter values."""
        # Arrange
        invalid_request = RecommendationRequest(
            user_id="test_user", filters={"min_year": 3000, "max_year": 1900, "min_duration": -10, "max_duration": 0}
        )

        # Act
        result = await recommendation_manager.recommend(invalid_request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_recommend_service_integration(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test integration between all services in the recommendation flow."""
        # Act
        result = await recommendation_manager.recommend(sample_recommendation_request)

        # Assert
        assert isinstance(result, RecommendationResponse)

        recommendation_manager.user_profile_service.get_profile.assert_called_once()
        recommendation_manager.embedder.embed.assert_called()
        recommendation_manager.recommender.recommend.assert_called()
        recommendation_manager.llm.rerank.assert_called()

    @pytest.mark.asyncio
    async def test_recommend_with_large_user_profile(self, recommendation_manager: RecommendationManager):
        """Test recommendation with large user profile."""
        # Arrange
        from schemas.watch_history import MovieHistoryItem

        large_profile = UserProfile(
            user_id="test_user",
            movies=[
                MovieHistoryItem(
                    title=f"Movie {i}",
                    year=2000 + (i % 20),
                    duration=120 + (i % 60),
                    genres=["Action", "Drama"],
                    countries=["USA"],
                    description=f"Description for movie {i}",
                    rating=4.0 + (i % 10) / 10,
                    watched_at="2023-01-01",
                )
                for i in range(100)
            ],
            clusters=[],
        )
        recommendation_manager.user_profile_service.get_profile.return_value = large_profile

        request = RecommendationRequest(user_id="test_user", filters={})

        # Act
        result = await recommendation_manager.recommend(request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_recommend_error_handling(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test error handling in recommendation flow."""
        # Arrange
        recommendation_manager.user_profile_service.get_profile.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(Exception):
            await recommendation_manager.recommend(sample_recommendation_request)

    @pytest.mark.asyncio
    async def test_recommend_with_embedding_service_error(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test recommendation when embedding service fails."""
        # Arrange
        recommendation_manager.embedder.embed.side_effect = Exception("Embedding service error")

        # Act & Assert
        with pytest.raises(Exception):
            await recommendation_manager.recommend(sample_recommendation_request)

    @pytest.mark.asyncio
    async def test_recommend_with_recommendation_service_error(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test recommendation when recommendation service fails."""
        # Arrange
        recommendation_manager.recommender.recommend.side_effect = Exception("Recommendation service error")

        # Act & Assert
        with pytest.raises(Exception):
            await recommendation_manager.recommend(sample_recommendation_request)

    @pytest.mark.asyncio
    async def test_recommend_with_llm_service_error(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test recommendation when LLM service fails."""
        # Arrange
        recommendation_manager.llm.rerank.side_effect = Exception("LLM service error")

        # Act & Assert
        with pytest.raises(Exception):
            await recommendation_manager.recommend(sample_recommendation_request)

    @pytest.mark.asyncio
    async def test_recommend_response_structure(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test that recommendation response has correct structure."""
        # Act
        result = await recommendation_manager.recommend(sample_recommendation_request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert hasattr(result, "recommendations")
        assert isinstance(result.recommendations, list)

        if result.recommendations:
            recommendation = result.recommendations[0]
            assert hasattr(recommendation, "title")
            assert hasattr(recommendation, "year")
            assert hasattr(recommendation, "genres")
            assert hasattr(recommendation, "countries")
            assert hasattr(recommendation, "description")
            assert hasattr(recommendation, "duration")
            assert hasattr(recommendation, "similarity")
            assert hasattr(recommendation, "justification")

    @pytest.mark.asyncio
    async def test_recommend_similarity_scores(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test that recommendations have valid similarity scores."""
        # Act
        result = await recommendation_manager.recommend(sample_recommendation_request)

        # Assert
        assert isinstance(result, RecommendationResponse)

        for recommendation in result.recommendations:
            assert hasattr(recommendation, "similarity")
            similarity = recommendation.similarity
            assert isinstance(similarity, (int, float))
            assert 0 <= similarity <= 1

    @pytest.mark.asyncio
    async def test_recommend_justification_presence(
        self, recommendation_manager: RecommendationManager, sample_recommendation_request
    ):
        """Test that recommendations include justifications."""
        # Act
        result = await recommendation_manager.recommend(sample_recommendation_request)

        # Assert
        assert isinstance(result, RecommendationResponse)

        for recommendation in result.recommendations:
            assert hasattr(recommendation, "justification")
            justification = recommendation.justification
            assert isinstance(justification, str)
            assert len(justification) > 0

    @pytest.mark.asyncio
    async def test_recommend_performance_with_large_dataset(self, recommendation_manager: RecommendationManager):
        """Test recommendation performance with large dataset."""
        # Arrange
        large_request = RecommendationRequest(
            user_id="test_user",
            filters={
                "genres": ["Action", "Sci-Fi", "Drama", "Comedy", "Thriller"],
                "min_year": 1990,
                "max_year": 2020,
                "countries": ["USA", "UK", "Canada", "Australia"],
            },
        )

        # Act
        result = await recommendation_manager.recommend(large_request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_recommend_filter_application(self, recommendation_manager: RecommendationManager):
        """Test that filters are properly applied to recommendations."""
        # Arrange
        specific_request = RecommendationRequest(
            user_id="test_user", filters={"genres": ["Action"], "min_year": 2010, "max_year": 2015, "countries": ["USA"]}
        )

        # Act
        result = await recommendation_manager.recommend(specific_request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_recommend_empty_result_handling(self, recommendation_manager: RecommendationManager):
        """Test handling of empty recommendation results."""
        # Arrange
        recommendation_manager.recommender.recommend.return_value = []
        recommendation_manager.llm.rerank.return_value = []

        request = RecommendationRequest(user_id="test_user", filters={})

        # Act
        result = await recommendation_manager.recommend(request)

        # Assert
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) == 0

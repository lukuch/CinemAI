from structlog.stdlib import BoundLogger

from domain.entities import Movie, UserProfile
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
from schemas.recommendation import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from services.user_profile_service import UserProfileService


class RecommendationManager:
    def __init__(
        self,
        embedder: EmbeddingService,
        clusterer: ClusteringService,
        tmdb: TMDBService,
        filterer: FilteringService,
        recommender: RecommendationService,
        field_detector: FieldDetectionService,
        llm: LLMService,
        vectorstore: VectorStoreRepository,
        user_profile_service: UserProfileService,
        logger: BoundLogger,
    ):
        self.embedder = embedder
        self.clusterer = clusterer
        self.tmdb = tmdb
        self.filterer = filterer
        self.recommender = recommender
        self.field_detector = field_detector
        self.llm = llm
        self.vectorstore = vectorstore
        self.user_profile_service = user_profile_service
        self.logger = logger

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """Main entry point: generate recommendations for a user."""
        user_id = request.user_id or "demo"
        filters = request.filters or {}
        self.logger.info(
            "Starting recommendation process",
            user_id=user_id,
            filters_applied=list(filters.keys()) if filters else "none",
        )

        user_profile = await self.user_profile_service.get_profile(user_id)
        if not user_profile:
            self.logger.error("User profile not found", user_id=user_id)
            raise ValueError(
                "User profile not found. Please upload your watch history first."
            )

        self.logger.info(
            "User profile loaded",
            user_id=user_id,
            total_movies=len(user_profile.movies) if user_profile.movies else 0,
            clusters_count=len(user_profile.clusters) if user_profile.clusters else 0,
            avg_rating=(
                round(
                    sum(m.rating for m in user_profile.movies)
                    / len(user_profile.movies),
                    3,
                )
                if user_profile.movies
                else 0
            ),
        )

        candidates = await self._fetch_candidates(request.filters)
        self.logger.info("Candidates fetched", count=len(candidates))

        filtered_candidates = await self._filter_and_embed_candidates(
            candidates, user_profile
        )
        self.logger.info(
            "Candidates filtered and embedded",
            original_count=len(candidates),
            filtered_count=len(filtered_candidates),
            reduction_rate=(
                round((len(candidates) - len(filtered_candidates)) / len(candidates), 3)
                if candidates
                else 0
            ),
        )

        recommendations = await self._generate_recommendations(
            user_profile, filtered_candidates
        )
        self.logger.info(
            "Recommendations generated",
            count=len(recommendations),
            avg_similarity=(
                round(
                    sum(r.similarity for r in recommendations) / len(recommendations), 3
                )
                if recommendations
                else 0
            ),
        )

        self.logger.info(
            "Recommendation process completed successfully",
            user_id=user_id,
            total_time_ms="TODO",  # Could add timing
            final_recommendations=len(recommendations),
        )

        return RecommendationResponse(recommendations=recommendations)

    async def _fetch_candidates(self, filters: dict) -> list:
        """Fetch movie candidates from TMDB API."""
        filters = filters or {}
        self.logger.info("Fetching movie candidates from TMDB", filters=filters)
        movies = await self.tmdb.fetch_movies(filters)
        self.logger.info("Fetched movie candidates from TMDB", count=len(movies))
        return movies

    async def _filter_and_embed_candidates(
        self, candidates: list, user_profile: UserProfile
    ) -> list:
        """Filter candidates and add embeddings."""
        self.logger.info("Filtering candidates", candidate_count=len(candidates))
        watched_movies = [
            Movie(
                id=getattr(m, "id", None),
                title=getattr(m, "title", ""),
                year=getattr(m, "year", None),
                duration=getattr(m, "duration", None),
                genres=getattr(m, "genres", []),
                countries=getattr(m, "countries", []),
                description=getattr(m, "description", None),
                embedding=None,
            )
            for m in (user_profile.movies or [])
        ]
        filters = {"watched_movies": watched_movies}
        filtered = self.filterer.filter(candidates, filters)
        self.logger.info("Filtered candidates", filtered_count=len(filtered))
        await self._add_embeddings_to_movies(filtered)
        self.logger.info("Added embeddings to filtered candidates", count=len(filtered))
        return filtered

    async def _add_embeddings_to_movies(self, movies: list):
        """Add embeddings to movie objects."""
        self.logger.info("Embedding movies", count=len(movies))
        texts = self.embedder.create_movie_texts(movies)
        embeddings = await self.embedder.embed(texts)
        for movie, embedding in zip(movies, embeddings):
            movie.embedding = embedding
        self.logger.info("Embeddings added to movies", count=len(movies))

    async def _generate_recommendations(
        self, user_profile: UserProfile, candidates: list
    ) -> list[RecommendationItem]:
        """Generate final recommendations with similarity scores and justifications."""
        self.logger.info("Generating recommendations", candidate_count=len(candidates))
        top_candidates_with_sim = self.recommender.recommend(user_profile, candidates)
        self.logger.info("Top candidates selected", count=len(top_candidates_with_sim))
        reranked = self.llm.rerank(
            user_profile, [m for _, m in top_candidates_with_sim]
        )
        self.logger.info("Candidates reranked with LLM", reranked_count=len(reranked))
        recommendations = []
        for i, (sim, movie) in enumerate(top_candidates_with_sim):
            justification = (
                reranked[i]["justification"] if reranked and i < len(reranked) else None
            )
            recommendations.append(
                self._create_recommendation_item(movie, sim, justification)
            )
        self.logger.info("Final recommendations created", count=len(recommendations))
        return recommendations

    def _create_recommendation_item(
        self, movie, similarity: float, justification: str
    ) -> RecommendationItem:
        """Create a recommendation item from movie data."""
        return RecommendationItem(
            title=movie.title,
            year=movie.year,
            genres=movie.genres,
            countries=movie.countries,
            description=movie.description,
            duration=movie.duration,
            similarity=round(similarity, 2),
            justification=justification,
        )

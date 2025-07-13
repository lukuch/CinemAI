import json

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
from schemas.recommendation import RecommendationItem, RecommendationRequest, RecommendationResponse
from schemas.watch_history import MovieHistoryItem


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
        self.logger = logger

    async def recommend(self, request: RecommendationRequest, watch_history_content: str = None) -> RecommendationResponse:
        """Main entry point: generate recommendations for a user."""
        user_id = request.user_id or "demo"
        self.logger.info("Starting recommendation process", user_id=user_id)
        user_profile = await self._get_or_create_user_profile(user_id, watch_history_content)
        self.logger.info("User profile ready", user_id=user_id, movies=len(user_profile.movies) if user_profile.movies else 0)
        candidates = await self._fetch_candidates(request.filters)
        self.logger.info("Fetched candidates", count=len(candidates))
        filtered_candidates = await self._filter_and_embed_candidates(candidates, user_profile)
        self.logger.info("Filtered and embedded candidates", count=len(filtered_candidates))
        recommendations = await self._generate_recommendations(user_profile, filtered_candidates)
        self.logger.info("Generated recommendations", count=len(recommendations))
        self.logger.info("Recommendation process complete", user_id=user_id)
        return RecommendationResponse(recommendations=recommendations)

    async def _get_or_create_user_profile(self, user_id: str, watch_history_content: str = None) -> UserProfile:
        """Get existing user profile or create one from watch history data."""
        try:
            user_profile = await self.vectorstore.get_user_profile(user_id)
            if user_profile:
                self.logger.info("Loaded user profile from vectorstore", user_id=user_id)
                return user_profile
        except Exception as e:
            self.logger.error("Error loading user profile", user_id=user_id, error=str(e))
        self.logger.info("Creating user profile from watch history", user_id=user_id)
        return await self._create_user_profile_from_history(user_id, watch_history_content)

    async def _create_user_profile_from_history(self, user_id: str, watch_history_content: str = None) -> UserProfile:
        """Create user profile from watch history data."""
        if not watch_history_content:
            self.logger.error("No watch history provided for user profile creation", user_id=user_id)
            raise ValueError("No watch history provided. Please upload a file.")
        watch_history = self._load_watch_history_from_content(watch_history_content)
        self.logger.info("Loaded watch history from content", user_id=user_id, count=len(watch_history))
        user_profile = await self._build_user_profile_from_history(user_id, watch_history)
        await self._save_user_profile(user_profile)
        self.logger.info("Created and saved user profile", user_id=user_id)
        return user_profile

    def _load_watch_history_from_content(self, content: str) -> list[MovieHistoryItem]:
        """
        Load watch history from content with pattern-based format detection.
        Supports both a top-level list and a dict with a 'movies' key.
        Args:
            content: JSON content as string
        """
        data = json.loads(content)
        if isinstance(data, list):
            movies_data = data
        elif isinstance(data, dict) and "movies" in data:
            movies_data = data["movies"]
        else:
            raise ValueError("File must be a list of movies or contain a 'movies' array")
        movies = []
        for movie_data in movies_data:
            try:
                # Use pattern matching for format detection
                converted_data = self.field_detector.convert_movie_data(movie_data)
                if converted_data is None:
                    self.logger.warning("Could not convert movie data", movie_data=movie_data)
                    continue
                # Validate the converted data
                if not self.field_detector.validate_movie_data(converted_data):
                    self.logger.warning("Invalid movie data after conversion", movie_data=movie_data)
                    continue
                movie_item = MovieHistoryItem(**converted_data)
                movies.append(movie_item)
            except Exception as e:
                self.logger.warning("Failed to process movie data", movie_data=movie_data, error=str(e))
                continue
        self.logger.info("Watch history loaded and validated", count=len(movies))
        return movies

    async def _build_user_profile_from_history(self, user_id: str, watch_history: list[MovieHistoryItem]) -> UserProfile:
        """Build user profile from watch history using embeddings and clustering."""
        high_rated_movies = [m for m in watch_history if m.rating > 4]
        self.logger.info("Building user profile from high-rated movies", user_id=user_id, count=len(high_rated_movies))
        texts = self._create_movie_texts(high_rated_movies)
        embeddings = await self.embedder.embed(texts)
        ratings = [m.rating for m in high_rated_movies]
        dates = [m.watched_at or "2023-01-01" for m in high_rated_movies]
        clusters = self.clusterer.cluster(embeddings, ratings, dates)
        self.logger.info("User profile built with clusters", user_id=user_id, clusters=len(clusters))
        return UserProfile(user_id=user_id, clusters=clusters, movies=watch_history)

    def _create_movie_texts(self, movies: list[MovieHistoryItem]) -> list[str]:
        """Create text representations of movies for embedding."""
        return [f"{m.title} {m.description or ''} {' '.join(m.genres)} {' '.join(m.countries)}" for m in movies]

    async def _save_user_profile(self, user_profile: UserProfile):
        """Save user profile to database."""
        try:
            await self.vectorstore.save_user_profile(user_profile)
            self.logger.info("User profile saved to vectorstore", user_id=user_profile.user_id)
        except Exception as e:
            self.logger.error("Error in save_user_profile", error=str(e))

    async def _fetch_candidates(self, filters: dict) -> list:
        """Fetch movie candidates from TMDB API."""
        filters = filters or {}
        self.logger.info("Fetching movie candidates from TMDB", filters=filters)
        movies = await self.tmdb.fetch_movies(filters)
        self.logger.info("Fetched movie candidates from TMDB", count=len(movies))
        return movies

    async def _filter_and_embed_candidates(self, candidates: list, user_profile: UserProfile) -> list:
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
        texts = self._create_movie_texts(movies)
        embeddings = await self.embedder.embed(texts)
        for movie, embedding in zip(movies, embeddings):
            movie.embedding = embedding
        self.logger.info("Embeddings added to movies", count=len(movies))

    async def _generate_recommendations(self, user_profile: UserProfile, candidates: list) -> list[RecommendationItem]:
        """Generate final recommendations with similarity scores and justifications."""
        self.logger.info("Generating recommendations", candidate_count=len(candidates))
        top_candidates_with_sim = self.recommender.recommend(user_profile, candidates)
        self.logger.info("Top candidates selected", count=len(top_candidates_with_sim))
        reranked = self.llm.rerank(user_profile, [m for _, m in top_candidates_with_sim])
        self.logger.info("Candidates reranked with LLM", reranked_count=len(reranked))
        recommendations = []
        for i, (sim, movie) in enumerate(top_candidates_with_sim):
            justification = reranked[i]["justification"] if reranked and i < len(reranked) else None
            recommendations.append(self._create_recommendation_item(movie, sim, justification))
        self.logger.info("Final recommendations created", count=len(recommendations))
        return recommendations

    def _create_recommendation_item(self, movie, similarity: float, justification: str) -> RecommendationItem:
        """Create a recommendation item from movie data."""
        return RecommendationItem(
            title=movie.title,
            year=movie.year,
            genres=movie.genres,
            countries=movie.countries,
            description=movie.description,
            similarity=round(similarity, 2),
            justification=justification,
        )

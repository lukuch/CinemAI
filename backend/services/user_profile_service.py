from structlog.stdlib import BoundLogger

from domain.entities import Movie, MovieHistoryItem, UserProfile
from domain.interfaces import (
    ClusteringService,
    EmbeddingService,
    FieldDetectionService,
    TMDBService,
    VectorStoreRepository,
)


class UserProfileService:
    def __init__(
        self,
        embedder: EmbeddingService,
        clusterer: ClusteringService,
        tmdb: TMDBService,
        field_detector: FieldDetectionService,
        vectorstore: VectorStoreRepository,
        logger: BoundLogger,
    ):
        self.embedder = embedder
        self.clusterer = clusterer
        self.tmdb = tmdb
        self.field_detector = field_detector
        self.vectorstore = vectorstore
        self.logger = logger

    async def get_profile(self, user_id: str) -> UserProfile:
        """Get user profile from database."""
        return await self.vectorstore.get_user_profile(user_id)

    async def create_and_save_profile(self, user_id: str, movies_data: list) -> UserProfile:
        """Create and save user profile from movies data."""
        self.logger.info(
            "Starting profile creation",
            user_id=user_id,
            input_movies=len(movies_data),
            avg_rating=sum(m.get("rating", 0) for m in movies_data) / len(movies_data) if movies_data else 0,
        )

        watch_history = await self._load_watch_history_from_content(movies_data)
        self.logger.info(
            "Watch history processed",
            user_id=user_id,
            valid_movies=len(watch_history),
            conversion_rate=len(watch_history) / len(movies_data) if movies_data else 0,
        )

        user_profile = await self._build_user_profile_from_history(user_id, watch_history)

        await self.vectorstore.save_user_profile(user_profile)
        self.logger.info(
            "Profile created and saved successfully",
            user_id=user_id,
            total_movies=len(user_profile.movies) if user_profile.movies else 0,
            clusters_created=len(user_profile.clusters) if user_profile.clusters else 0,
        )

        return user_profile

    async def _load_watch_history_from_content(self, movies_data: list) -> list[MovieHistoryItem]:
        """Load watch history from movies data with batch processing."""
        movies = []
        # Use batch processing for better performance
        converted_movies, excluded_movies = await self.field_detector.convert_movies_batch(movies_data)

        for excluded in excluded_movies:
            self.logger.warning(
                "Movie excluded after enrichment/validation",
                original=excluded["original"],
                enriched=excluded["enriched"],
                missing_fields=excluded["missing_fields"],
            )

        for converted_data in converted_movies:
            try:
                # Validate the converted data (extra safety)
                if not self.field_detector.validate_movie_data(converted_data):
                    self.logger.warning("Invalid movie data after conversion", movie_data=converted_data)
                    continue
                movie_item = MovieHistoryItem(**converted_data)
                movies.append(movie_item)
            except Exception as e:
                self.logger.warning("Failed to create movie item", movie_data=converted_data, error=str(e))
                continue
        self.logger.info("Watch history loaded and validated", count=len(movies))
        return movies

    async def _build_user_profile_from_history(self, user_id: str, watch_history: list[MovieHistoryItem]) -> UserProfile:
        self.logger.info("Building user profile from history", user_id=user_id, total_movies=len(watch_history))

        high_rated_movies = [m for m in watch_history if m.rating > 4]
        self.logger.info(
            "High-rated movies filtered",
            high_rated_count=len(high_rated_movies),
            total_movies=len(watch_history),
            avg_rating=sum(m.rating for m in watch_history) / len(watch_history) if watch_history else 0,
        )

        texts = [f"{m.title} {m.description or ''} {' '.join(m.genres)} {' '.join(m.countries)}" for m in high_rated_movies]
        self.logger.info("Text preparation completed", texts_count=len(texts))

        embeddings = await self.embedder.embed(texts)
        self.logger.info("Embeddings generated", embeddings_count=len(embeddings))

        ratings = [m.rating for m in high_rated_movies]
        dates = [m.watched_at or "2023-01-01" for m in high_rated_movies]
        movies = [
            Movie(
                id=getattr(m, "id", None),
                title=m.title,
                year=m.year,
                duration=m.duration,
                genres=m.genres,
                countries=m.countries,
                description=m.description,
                embedding=embeddings[i] if i < len(embeddings) else None,
            )
            for i, m in enumerate(high_rated_movies)
        ]

        clusters = self.clusterer.cluster(embeddings, ratings, dates, movies)
        self.logger.info(
            "User profile built successfully",
            user_id=user_id,
            clusters_count=len(clusters),
            total_movies=len(watch_history),
            high_rated_movies=len(high_rated_movies),
        )

        return UserProfile(user_id=user_id, clusters=clusters, movies=watch_history)

from schemas.watch_history import MovieHistoryItem
from schemas.recommendation import RecommendationRequest, RecommendationResponse, RecommendationItem
from services.embedding_service import OpenAIEmbeddingService
from services.clustering_service import SklearnClusteringService
from services.tmdb_service import TMDBApiService
from services.filtering_service import DefaultFilteringService
from services.recommendation_service import DefaultRecommendationService
from services.llm_service import OpenAILLMService
from repositories.vector_store import PgvectorRepository
import os
import json
from domain.entities import UserProfile

class RecommendationManager:
    def __init__(self,
                 embedder: OpenAIEmbeddingService,
                 clusterer: SklearnClusteringService,
                 tmdb: TMDBApiService,
                 filterer: DefaultFilteringService,
                 recommender: DefaultRecommendationService,
                 llm: OpenAILLMService,
                 vectorstore: PgvectorRepository):
        self.embedder = embedder
        self.clusterer = clusterer
        self.tmdb = tmdb
        self.filterer = filterer
        self.recommender = recommender
        self.llm = llm
        self.vectorstore = vectorstore

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        user_id = request.user_id or "demo"
        user_profile = await self._get_or_create_user_profile(user_id)
        candidates = await self._fetch_candidates(request.filters)
        filtered_candidates = await self._filter_and_embed_candidates(candidates, user_profile)
        recommendations = await self._generate_recommendations(user_profile, filtered_candidates)
        return RecommendationResponse(recommendations=recommendations)

    async def _get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """Get existing user profile or create one from demo data."""
        user_profile = await self._try_get_user_profile(user_id)
        if not user_profile:
            user_profile = await self._create_user_profile_from_demo_data(user_id)
        return user_profile

    async def _try_get_user_profile(self, user_id: str) -> UserProfile:
        """Attempt to retrieve user profile from database."""
        try:
            return await self.vectorstore.get_user_profile(user_id)
        except Exception:
            return None

    async def _create_user_profile_from_demo_data(self, user_id: str) -> UserProfile:
        """Create user profile from demo watch history data."""
        watch_history = self._load_demo_watch_history()
        user_profile = await self._build_user_profile_from_history(user_id, watch_history)
        await self._save_user_profile(user_profile)
        return user_profile

    def _load_demo_watch_history(self) -> list[MovieHistoryItem]:
        """Load demo watch history from JSON file."""
        demo_path = os.path.join(os.path.dirname(__file__), '../demo-data/demo_watch_history.json')
        with open(demo_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [MovieHistoryItem(**m) for m in data["movies"]]

    async def _build_user_profile_from_history(self, user_id: str, watch_history: list[MovieHistoryItem]) -> UserProfile:
        """Build user profile from watch history using embeddings and clustering."""
        high_rated_movies = [m for m in watch_history if m.rating > 4]
        texts = self._create_movie_texts(high_rated_movies)
        embeddings = await self.embedder.embed(texts)
        ratings = [m.rating for m in high_rated_movies]
        dates = [m.watched_at or "2023-01-01" for m in high_rated_movies]
        clusters = self.clusterer.cluster(embeddings, ratings, dates)
        return UserProfile(user_id=user_id, clusters=clusters)

    def _create_movie_texts(self, movies: list[MovieHistoryItem]) -> list[str]:
        """Create text representations of movies for embedding."""
        return [f"{m.title} {m.description or ''} {' '.join(m.genres)} {' '.join(m.countries)}" for m in movies]

    async def _save_user_profile(self, user_profile: UserProfile):
        """Save user profile to database."""
        try:
            await self.vectorstore.save_user_profile(user_profile)
        except Exception as e:
            print(f"Error in save_user_profile: {e}")

    async def _fetch_candidates(self, filters: dict) -> list:
        """Fetch movie candidates from TMDB API."""
        filters = filters or {}
        return await self.tmdb.fetch_movies(filters)

    async def _filter_and_embed_candidates(self, candidates: list, user_profile: UserProfile) -> list:
        """Filter candidates and add embeddings."""
        watched_titles = set([m.title for m in getattr(user_profile, 'movies', [])])
        filters = {"watched_ids": watched_titles}
        filtered = self.filterer.filter(candidates, filters)
        await self._add_embeddings_to_movies(filtered)
        return filtered

    async def _add_embeddings_to_movies(self, movies: list):
        """Add embeddings to movie objects."""
        texts = self._create_movie_texts(movies)
        embeddings = await self.embedder.embed(texts)
        for movie, embedding in zip(movies, embeddings):
            movie.embedding = embedding

    async def _generate_recommendations(self, user_profile: UserProfile, candidates: list) -> list[RecommendationItem]:
        """Generate final recommendations with similarity scores and justifications."""
        top_candidates_with_sim = self.recommender.recommend(user_profile, candidates)
        reranked = self.llm.rerank(user_profile, [m for _, m in top_candidates_with_sim])
        
        recommendations = []
        for i, (sim, movie) in enumerate(top_candidates_with_sim):
            justification = reranked[i]["justification"] if reranked and i < len(reranked) else None
            recommendations.append(self._create_recommendation_item(movie, sim, justification))
        
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
            justification=justification
        ) 
from schemas.watch_history import MovieHistoryItem
from schemas.recommendation import RecommendationRequest, RecommendationResponse, RecommendationItem
from services.embedding_service import OpenAIEmbeddingService
from services.clustering_service import SklearnClusteringService
from services.tmdb_service import TMDBApiService
from services.filtering_service import DefaultFilteringService
from services.recommendation_service import DefaultRecommendationService
from services.llm_service import OpenAILLMService
from repositories.vector_store import PgvectorRepository
from sqlalchemy.ext.asyncio import AsyncSession
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
                 session: AsyncSession):
        self.embedder = embedder
        self.clusterer = clusterer
        self.tmdb = tmdb
        self.filterer = filterer
        self.recommender = recommender
        self.llm = llm
        self.session = session
        self.vectorstore = PgvectorRepository(session)

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        user_id = request.user_id or "demo"
        user_profile = None
        try:
            user_profile = await self.vectorstore.get_user_profile(user_id)
        except Exception:
            user_profile = None
        if not user_profile:
            demo_path = os.path.join(os.path.dirname(__file__), '../demo-data/demo_watch_history.json')
            with open(demo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                watch_history = [MovieHistoryItem(**m) for m in data["movies"]]
            texts = [f"{m.title} {m.description or ''} {' '.join(m.genres)} {' '.join(m.countries)}" for m in watch_history if m.rating > 4]
            embeddings = self.embedder.embed(texts)
            ratings = [m.rating for m in watch_history if m.rating > 4]
            dates = [m.watched_at or "2023-01-01" for m in watch_history if m.rating > 4]
            clusters = self.clusterer.cluster(embeddings, ratings, dates)
            user_profile = UserProfile(user_id=user_id, clusters=clusters)
            try:
                await self.vectorstore.save_user_profile(user_profile)
            except Exception:
                pass
        filters = request.filters or {}
        candidates = self.tmdb.fetch_movies(filters)
        watched_titles = set([m.title for m in getattr(user_profile, 'movies', [])])
        filters["watched_ids"] = watched_titles
        filtered = self.filterer.filter(candidates, filters)
        texts = [f"{m.title} {m.description or ''} {' '.join(m.genres)} {' '.join(m.countries)}" for m in filtered]
        candidate_embeddings = self.embedder.embed(texts)
        for m, emb in zip(filtered, candidate_embeddings):
            m.embedding = emb
        top_candidates = self.recommender.recommend(user_profile, filtered)
        reranked = self.llm.rerank(user_profile, top_candidates)
        recs = []
        for i, m in enumerate(top_candidates):
            recs.append(RecommendationItem(
                title=m.title,
                year=m.year,
                genres=m.genres,
                countries=m.countries,
                description=m.description,
                similarity=0.0,
                justification=reranked[0]["text"] if reranked and i == 0 else None
            ))
        return RecommendationResponse(recommendations=recs) 
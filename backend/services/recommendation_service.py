from domain.interfaces import RecommendationService
from domain.entities import UserProfile, Movie
from typing import List
from utils.similarity import cosine_similarity

class DefaultRecommendationService(RecommendationService):
    def recommend(self, user_profile: UserProfile, candidates: List[Movie]) -> List[Movie]:
        scored = []
        for movie in candidates:
            if not movie.embedding:
                continue
            max_sim = max(
                cosine_similarity(movie.embedding.vector, c.centroid.vector)
                for c in user_profile.clusters
            )
            scored.append((max_sim, movie))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored[:10]] 
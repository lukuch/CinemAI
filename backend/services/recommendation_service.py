from domain.interfaces import RecommendationService
from domain.entities import UserProfile, Movie
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
import numpy as np

def cosine_similarity(a, b):
    # a and b are 1D lists or arrays
    return sk_cosine_similarity([a], [b])[0][0]

class DefaultRecommendationService(RecommendationService):
    def recommend(self, user_profile: UserProfile, candidates: List[Movie]) -> List[Tuple[float, Movie]]:
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
        return scored[:10] 
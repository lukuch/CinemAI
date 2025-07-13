from typing import List, Tuple

import numpy as np

from domain.entities import Movie, UserProfile
from domain.interfaces import RecommendationService


class DefaultRecommendationService(RecommendationService):
    def recommend(self, user_profile: UserProfile, candidates: List[Movie]) -> List[Tuple[float, Movie]]:
        # Stack all cluster centroids into a matrix
        centroids = np.stack([c.centroid.vector for c in user_profile.clusters])
        centroids_norm = np.linalg.norm(centroids, axis=1)
        scored = []
        for movie in candidates:
            if not movie.embedding:
                continue
            movie_vec = np.array(movie.embedding.vector)
            movie_norm = np.linalg.norm(movie_vec)
            # Compute cosine similarity to all centroids at once
            sims = centroids @ movie_vec / (centroids_norm * movie_norm + 1e-8)
            max_sim = np.max(sims)
            scored.append((max_sim, movie))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:10]

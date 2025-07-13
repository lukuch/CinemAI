from typing import List, Tuple

import numpy as np
from injector import inject
from structlog.stdlib import BoundLogger

from domain.entities import Movie, UserProfile
from domain.interfaces import RecommendationService


class DefaultRecommendationService(RecommendationService):
    @inject
    def __init__(self, logger: BoundLogger):
        self.logger = logger

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
        top_recommendations = scored[:10]

        if top_recommendations:
            similarities = [sim for sim, _ in top_recommendations]
            self.logger.info(
                "Recommendation scoring completed",
                total_candidates=len(candidates),
                candidates_with_embeddings=len(scored),
                top_similarity=round(max(similarities), 3),
                avg_similarity=round(sum(similarities) / len(similarities), 3),
                min_similarity=round(min(similarities), 3),
            )

        return top_recommendations

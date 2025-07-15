from typing import List, Tuple

import numpy as np
from injector import inject
from structlog.stdlib import BoundLogger

from domain.entities import Movie, UserProfile
from domain.interfaces import IRecommendationService


class RecommendationService(IRecommendationService):
    @inject
    def __init__(self, logger: BoundLogger):
        self.logger = logger

    def recommend(
        self, user_profile: UserProfile, candidates: List[Movie], alpha: float = 5.0
    ) -> List[Tuple[float, Movie]]:
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
            # Softmax aggregation
            exp_sims = np.exp(alpha * sims)
            weights = exp_sims / (np.sum(exp_sims) + 1e-8)
            softmax_score = np.sum(weights * sims)
            scored.append((softmax_score, movie))
        scored.sort(reverse=True, key=lambda x: x[0])
        top_recommendations = scored[:10]

        if top_recommendations:
            similarities = [sim for sim, _ in top_recommendations]
            self.logger.info(
                "Recommendation scoring completed (softmax aggregation)",
                total_candidates=len(candidates),
                candidates_with_embeddings=len(scored),
                top_similarity=round(max(similarities), 3),
                avg_similarity=round(sum(similarities) / len(similarities), 3),
                min_similarity=round(min(similarities), 3),
                aggregation_method="softmax",
                softmax_alpha=alpha,
            )

        return top_recommendations

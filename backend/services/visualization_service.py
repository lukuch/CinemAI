from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import orjson
import umap
from injector import inject
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from domain.entities import Embedding
from domain.interfaces import ICacheRepository, IEmbeddingService, IVisualizationService


class VisualizationService(IVisualizationService):
    SUPPORTED_METHODS = ("tsne", "umap", "pca")

    @inject
    def __init__(self, embedder: IEmbeddingService, redis: ICacheRepository) -> None:
        self.embedder = embedder
        self.redis = redis

    async def get_user_profile_visualization(
        self, profile: Any, method: str = "tsne"
    ) -> Dict[str, Any]:
        """
        Generate a 2D visualization of user profile clusters and movies.
        Args:
            profile: User profile object with clusters and movies.
            method: Dimensionality reduction method ('tsne', 'umap', 'pca').
        Returns:
            Dict with visualization points.
        """
        movies_to_embed = self._extract_unique_movies(profile)
        texts = self.embedder.create_movie_texts(movies_to_embed)
        embeddings, movie_titles = await self._get_movie_embeddings(
            texts, movies_to_embed
        )

        centroids = np.array([c.centroid.vector for c in profile.clusters])
        X = self._stack_centroids_and_embeddings(centroids, embeddings)
        labels = [f"Cluster {i+1}" for i in range(len(centroids))] + movie_titles
        point_types = ["cluster"] * len(centroids) + ["movie"] * len(embeddings)

        coords = self._reduce_dimensions(X, method)
        data = self._build_points_data(
            coords, labels, point_types, movies_to_embed, len(centroids)
        )
        return {"points": data}

    def _extract_unique_movies(self, profile: Any) -> List[Any]:
        seen_titles = set()
        unique_movies = []
        for cluster in getattr(profile, "clusters", []):
            for movie in getattr(cluster, "movies", []) or []:
                if movie.title not in seen_titles:
                    unique_movies.append(movie)
                    seen_titles.add(movie.title)
        return unique_movies

    async def _get_movie_embeddings(
        self, texts: Sequence[str], movies: Sequence[Any]
    ) -> Tuple[List[Any], List[str]]:
        if not texts:
            return [], []
        embeddings = []
        for text in texts:
            key = self.embedder._cache_key(text)
            cached = self.redis.get(key)
            if cached:
                vector = orjson.loads(cached)
                embeddings.append(Embedding(vector=vector))
            else:
                embedding = await self.embedder.embed([text])
                if embedding and embedding[0] is not None:
                    embeddings.append(embedding[0])
                    self.redis.set(
                        key, orjson.dumps(embedding[0].vector), expire=60 * 60 * 24 * 30
                    )
                else:
                    embeddings.append(None)
        filtered_embeddings = [e for e in embeddings if e is not None]
        filtered_titles = [m.title for m, e in zip(movies, embeddings) if e is not None]
        return filtered_embeddings, filtered_titles

    def _stack_centroids_and_embeddings(
        self, centroids: np.ndarray, embeddings: List[Any]
    ) -> np.ndarray:
        if embeddings:
            movie_embeddings = np.array([e.vector for e in embeddings])
            return np.vstack([centroids, movie_embeddings])
        return centroids

    def _reduce_dimensions(self, X: np.ndarray, method: str) -> List[List[float]]:
        if X.shape[0] < 3:
            return X.tolist()
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError(
                f"Unknown visualization method: {method}. Supported: {self.SUPPORTED_METHODS}"
            )
        return reducer.fit_transform(X).tolist()

    def _build_points_data(
        self,
        coords: List[List[float]],
        labels: List[str],
        point_types: List[str],
        movies: List[Any],
        n_centroids: int,
    ) -> List[Dict[str, Any]]:
        data = []
        for i, (coord, label, point_type) in enumerate(
            zip(coords, labels, point_types)
        ):
            point = {"x": coord[0], "y": coord[1], "label": label, "type": point_type}
            if point_type == "movie":
                movie = movies[i - n_centroids]  # offset for clusters
                point["year"] = getattr(movie, "year", None)
                point["genres"] = getattr(movie, "genres", None)
                point["rating"] = getattr(movie, "rating", None)
                point["duration"] = getattr(movie, "duration", None)
            data.append(point)
        return data

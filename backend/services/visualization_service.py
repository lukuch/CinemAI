import numpy as np
import umap
from injector import inject
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from domain.interfaces import EmbeddingService


class VisualizationService:
    @inject
    def __init__(self, embedder: EmbeddingService):
        self.embedder = embedder
        self._embedding_cache = {}  # In-memory cache for embeddings

    async def get_user_profile_visualization(self, profile, method: str = "tsne"):
        all_movies = []
        seen_titles = set()
        for cluster in profile.clusters:
            for movie in getattr(cluster, "movies", []) or []:
                if movie.title not in seen_titles:
                    all_movies.append(movie)
                    seen_titles.add(movie.title)
        movies_to_embed = all_movies

        texts = self.embedder.create_movie_texts(movies_to_embed)
        if texts:
            cache_key = tuple(texts)
            if cache_key in self._embedding_cache:
                embeddings = self._embedding_cache[cache_key]
            else:
                embeddings = await self.embedder.embed(texts)
                self._embedding_cache[cache_key] = embeddings
            movie_embeddings = np.array([e.vector for e in embeddings if e is not None])
            movie_titles = [
                m.title for m, e in zip(movies_to_embed, embeddings) if e is not None
            ]
        else:
            movie_embeddings = np.zeros((0, 2))
            movie_titles = []

        centroids = np.array([c.centroid.vector for c in profile.clusters])
        X = (
            np.vstack([centroids, movie_embeddings])
            if len(movie_embeddings) > 0
            else centroids
        )
        labels = [f"Cluster {i+1}" for i in range(len(centroids))] + movie_titles
        types = ["cluster" for _ in range(len(centroids))] + [
            "movie" for _ in range(len(movie_embeddings))
        ]

        if X.shape[0] < 3:
            coords = X.tolist()
        else:
            if method == "tsne":
                reducer = TSNE(n_components=2, random_state=42)
                coords = reducer.fit_transform(X).tolist()
            elif method == "umap":
                reducer = umap.UMAP(n_components=2, random_state=42)
                coords = reducer.fit_transform(X).tolist()
            elif method == "pca":
                reducer = PCA(n_components=2)
                coords = reducer.fit_transform(X).tolist()
            else:
                raise ValueError(f"Unknown visualization method: {method}")

        data = [
            {"x": coord[0], "y": coord[1], "label": label, "type": typ}
            for coord, label, typ in zip(coords, labels, types)
        ]
        return {"points": data}

from datetime import datetime
from typing import List

import hdbscan
import numpy as np
from injector import inject
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from structlog.stdlib import BoundLogger

from domain.entities import Cluster, Embedding, Movie
from domain.interfaces import IClusteringService
from utils.weighting import rating_weight, recency_weight


class SklearnClusteringService(IClusteringService):
    @inject
    def __init__(self, logger: BoundLogger):
        self.logger = logger

    def _kmeans_clusters(self, X, weights, ratings, n_clusters, movies):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        clusters = []
        for i in range(n_clusters):
            idx = np.where(labels == i)[0]
            if len(idx) == 0:
                continue
            centroid = np.average(X[idx], axis=0, weights=weights[idx])
            clusters.append(
                Cluster(
                    centroid=Embedding(list(centroid)),
                    movies=[movies[j] for j in idx],
                    average_rating=np.average(
                        np.array(ratings)[idx], weights=weights[idx]
                    ),
                    count=len(idx),
                )
            )
        return clusters

    def _best_kmeans_clusters(self, X, weights, ratings, movies, min_k=2, max_k=10):
        best_k = min_k
        best_score = -1
        n = len(X)
        for k in range(min_k, min(max_k, n)):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        clusters = self._kmeans_clusters(X, weights, ratings, best_k, movies)
        self.logger.info(
            "KMeans clustering completed",
            clusters=len(clusters),
            best_k=best_k,
            silhouette_score=round(best_score, 3),
        )
        return clusters

    def _hdbscan_clusters(
        self,
        X,
        weights,
        ratings,
        movies,
        min_cluster_size=10,
        min_samples=2,
        noise_threshold=0.5,
    ):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        labels = clusterer.fit_predict(X)
        unique, counts = np.unique(labels, return_counts=True)
        self.logger.info(
            "Cluster label distribution", distribution=dict(zip(unique, counts))
        )
        if sum(counts[unique == -1]) > noise_threshold * len(X):
            self.logger.warning("Too many noise points, falling back to KMeans")
            return None
        clusters = []
        for i in set(labels):
            if i == -1:
                continue
            idx = np.where(labels == i)[0]
            centroid = np.average(X[idx], axis=0, weights=weights[idx])
            clusters.append(
                Cluster(
                    centroid=Embedding(list(centroid)),
                    movies=[movies[j] for j in idx],
                    average_rating=np.average(
                        np.array(ratings)[idx], weights=weights[idx]
                    ),
                    count=len(idx),
                )
            )
        self.logger.info("HDBSCAN clustering completed", clusters=len(clusters))
        return clusters

    def cluster(
        self,
        embeddings: List[Embedding],
        ratings: List[float],
        dates: List[str],
        movies: List[Movie],
    ) -> List[Cluster]:
        self.logger.info(
            "Starting clustering process",
            embeddings_count=len(embeddings),
            movies_count=len(movies),
            avg_rating=round(sum(ratings) / len(ratings), 3) if ratings else 0,
        )

        X = np.array([e.vector for e in embeddings])
        weights = np.array(
            [
                rating_weight(r) * recency_weight(d, datetime.now())
                for r, d in zip(ratings, dates)
            ]
        )
        n = len(embeddings)

        if n < 100:
            self.logger.info(
                "Using single cluster approach (small dataset)", movies_count=n
            )
            centroid = np.average(X, axis=0, weights=weights)
            cluster = Cluster(
                centroid=Embedding(list(centroid)),
                movies=movies,
                average_rating=np.average(ratings, weights=weights),
                count=n,
            )
            self.logger.info(
                "Single cluster created",
                movies_count=n,
                avg_rating=round(cluster.average_rating, 3),
            )
            return [cluster]
        elif n <= 500:
            self.logger.info("Using KMeans clustering (medium dataset)", movies_count=n)
            return self._best_kmeans_clusters(X, weights, ratings, movies)
        else:
            self.logger.info("Using HDBSCAN clustering (large dataset)", movies_count=n)
            clusters = self._hdbscan_clusters(X, weights, ratings, movies)
            if clusters is None:
                self.logger.info("HDBSCAN failed, falling back to KMeans")
                return self._best_kmeans_clusters(X, weights, ratings, movies)
            return clusters

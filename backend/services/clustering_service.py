from datetime import datetime
from typing import List

import hdbscan
import numpy as np
from injector import inject
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from structlog.stdlib import BoundLogger

from domain.entities import Cluster, Embedding, Movie
from domain.interfaces import ClusteringService
from utils.weighting import rating_weight, recency_weight


class SklearnClusteringService(ClusteringService):
    @inject
    def __init__(self, logger: BoundLogger):
        self.logger = logger

    def _kmeans_clusters(self, X, weights, ratings, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
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
                    movies=[],
                    average_rating=np.average(np.array(ratings)[idx], weights=weights[idx]),
                    count=len(idx),
                )
            )
        return clusters

    def _best_kmeans_clusters(self, X, weights, ratings, min_k=2, max_k=10):
        best_k = min_k
        best_score = -1
        n = len(X)
        for k in range(min_k, min(max_k, n)):
            kmeans = KMeans(n_clusters=k, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        return self._kmeans_clusters(X, weights, ratings, best_k)

    def _hdbscan_clusters(self, X, weights, ratings, min_cluster_size=10, min_samples=2, noise_threshold=0.5):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(X)
        unique, counts = np.unique(labels, return_counts=True)
        self.logger.info("Cluster label distribution", distribution=dict(zip(unique, counts)))
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
                    movies=[],
                    average_rating=np.average(np.array(ratings)[idx], weights=weights[idx]),
                    count=len(idx),
                )
            )
        return clusters

    def cluster(self, embeddings: List[Embedding], ratings: List[float], dates: List[str]) -> List[Cluster]:
        X = np.array([e.vector for e in embeddings])
        weights = np.array([rating_weight(r) * recency_weight(d, datetime.now()) for r, d in zip(ratings, dates)])
        n = len(embeddings)
        if n < 100:
            centroid = np.average(X, axis=0, weights=weights)
            return [
                Cluster(
                    centroid=Embedding(list(centroid)), movies=[], average_rating=np.average(ratings, weights=weights), count=n
                )
            ]
        elif n <= 500:
            return self._best_kmeans_clusters(X, weights, ratings)
        else:
            clusters = self._hdbscan_clusters(X, weights, ratings)
            if clusters is None:
                return self._best_kmeans_clusters(X, weights, ratings)
            return clusters

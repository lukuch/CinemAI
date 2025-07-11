from domain.interfaces import ClusteringService
from domain.entities import Embedding, Cluster, Movie
from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
from datetime import datetime
from utils.weighting import rating_weight, recency_weight

class SklearnClusteringService(ClusteringService):
    def cluster(self, embeddings: List[Embedding], ratings: List[float], dates: List[str]) -> List[Cluster]:
        X = np.array([e.vector for e in embeddings])
        weights = np.array([
            rating_weight(r) * recency_weight(d, datetime.now())
            for r, d in zip(ratings, dates)
        ])
        n = len(embeddings)
        if n < 100:
            centroid = np.average(X, axis=0, weights=weights)
            return [Cluster(centroid=Embedding(list(centroid)), movies=[], average_rating=np.average(ratings, weights=weights), count=n)]
        elif n <= 500:
            best_k = 2
            best_score = -1
            for k in range(2, min(10, n)):
                kmeans = KMeans(n_clusters=k, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            kmeans = KMeans(n_clusters=best_k, n_init=10)
            labels = kmeans.fit_predict(X)
            clusters = []
            for i in range(best_k):
                idx = np.where(labels == i)[0]
                if len(idx) == 0:
                    continue
                centroid = np.average(X[idx], axis=0, weights=weights[idx])
                clusters.append(Cluster(centroid=Embedding(list(centroid)), movies=[], average_rating=np.average(np.array(ratings)[idx], weights=weights[idx]), count=len(idx)))
            return clusters
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
            labels = clusterer.fit_predict(X)
            clusters = []
            for i in set(labels):
                if i == -1:
                    continue
                idx = np.where(labels == i)[0]
                centroid = np.average(X[idx], axis=0, weights=weights[idx])
                clusters.append(Cluster(centroid=Embedding(list(centroid)), movies=[], average_rating=np.average(np.array(ratings)[idx], weights=weights[idx]), count=len(idx)))
            return clusters 
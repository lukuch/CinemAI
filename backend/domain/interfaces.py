from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .entities import Movie, UserProfile, Embedding, Cluster

class EmbeddingService(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[Embedding]:
        pass

class ClusteringService(ABC):
    @abstractmethod
    def cluster(self, embeddings: List[Embedding], ratings: List[float], dates: List[str]) -> List[Cluster]:
        pass

class TMDBService(ABC):
    @abstractmethod
    def fetch_movies(self, filters: Dict[str, Any]) -> List[Movie]:
        pass

class FilteringService(ABC):
    @abstractmethod
    def filter(self, movies: List[Movie], filters: Dict[str, Any]) -> List[Movie]:
        pass

class RecommendationService(ABC):
    @abstractmethod
    def recommend(self, user_profile: UserProfile, candidates: List[Movie]) -> List[Movie]:
        pass

class LLMService(ABC):
    @abstractmethod
    def rerank(self, user_profile: UserProfile, candidates: List[Movie]) -> List[Dict[str, Any]]:
        pass

class VectorStoreRepository(ABC):
    @abstractmethod
    def save_user_profile(self, profile: UserProfile):
        pass
    @abstractmethod
    def get_user_profile(self, user_id: str) -> UserProfile:
        pass

class CacheRepository(ABC):
    @abstractmethod
    def get(self, key: str):
        pass
    @abstractmethod
    def set(self, key: str, value: Any, expire: int = 3600):
        pass 
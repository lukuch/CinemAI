from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .entities import Cluster, Embedding, Movie, UserProfile


class IEmbeddingService(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[Embedding]:
        pass

    @abstractmethod
    def create_movie_texts(self, movies: list) -> list[str]:
        pass


class IClusteringService(ABC):
    @abstractmethod
    def cluster(
        self, embeddings: List[Embedding], ratings: List[float], dates: List[str]
    ) -> List[Cluster]:
        pass


class IMovieApiService(ABC):
    @abstractmethod
    def fetch_movies(self, filters: Dict[str, Any]) -> List[Movie]:
        pass

    @abstractmethod
    async def enrich_movies_batch(
        self, movies_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        pass


class IFilteringService(ABC):
    @abstractmethod
    def filter(self, movies: List[Movie], filters: Dict[str, Any]) -> List[Movie]:
        pass


class IRecommendationService(ABC):
    @abstractmethod
    def recommend(
        self, user_profile: UserProfile, candidates: List[Movie]
    ) -> List[Movie]:
        pass


class ILLMService(ABC):
    @abstractmethod
    def rerank(
        self, user_profile: UserProfile, candidates: List[Movie]
    ) -> List[Dict[str, Any]]:
        pass


class IVectorStoreRepository(ABC):
    @abstractmethod
    def save_user_profile(self, profile: UserProfile):
        pass

    @abstractmethod
    def get_user_profile(self, user_id: str) -> UserProfile:
        pass


class ICacheRepository(ABC):
    @abstractmethod
    def get(self, key: str):
        pass

    @abstractmethod
    def set(self, key: str, value: Any, expire: int = 3600):
        pass


class IFieldDetectionService(ABC):
    @abstractmethod
    async def convert_movies_batch(
        self, movies_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def validate_movie_data(self, data: dict) -> bool:
        pass

    @abstractmethod
    def normalize_title_value(self, title: str) -> str:
        pass

    @abstractmethod
    def map_country_to_tmdb(self, country: str) -> str:
        pass


class IVisualizationService(ABC):
    @abstractmethod
    async def get_user_profile_visualization(self, profile, method: str = "tsne"):
        pass


class IUserProfileService(ABC):
    @abstractmethod
    async def get_profile(self, user_id: str):
        pass

    # Add other abstract methods as needed

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel

from schemas.watch_history import MovieHistoryItem


@dataclass
class Embedding:
    vector: List[float]


@dataclass
class Movie:
    id: str
    title: str
    year: int
    duration: int
    genres: List[str]
    countries: List[str]
    description: Optional[str]
    embedding: Optional[Embedding] = None


@dataclass
class Cluster:
    centroid: Embedding
    movies: List[Movie]
    average_rating: float
    count: int


class UserProfile(BaseModel):
    user_id: str
    clusters: List[Cluster]
    movies: Optional[List[MovieHistoryItem]] = None

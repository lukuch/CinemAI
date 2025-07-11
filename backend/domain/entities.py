from dataclasses import dataclass
from typing import List, Optional

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

@dataclass
class UserProfile:
    user_id: str
    clusters: List[Cluster] 
from typing import List, Optional

from pydantic import BaseModel


class MovieHistoryItem(BaseModel):
    title: str
    rating: float
    year: int
    duration: int
    genres: List[str]
    countries: List[str]
    description: Optional[str]
    watched_at: Optional[str]


class WatchHistoryUpload(BaseModel):
    movies: List[MovieHistoryItem]

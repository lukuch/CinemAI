from typing import List, Optional

from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    user_id: Optional[str]
    filters: Optional[dict]


class RecommendationItem(BaseModel):
    title: str
    year: int
    genres: List[str]
    countries: List[str]
    description: Optional[str]
    similarity: float
    justification: Optional[str]


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]

from pydantic import BaseModel
from typing import List, Optional

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
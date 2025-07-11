from pydantic import BaseModel
from typing import List

class FiltersResponse(BaseModel):
    genres: List[str]
    years: List[int]
    durations: List[int]
    countries: List[str] 
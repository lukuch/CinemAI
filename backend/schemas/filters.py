from typing import List

from pydantic import BaseModel


class FiltersResponse(BaseModel):
    genres: List[str]
    years: List[int]
    durations: List[int]
    countries: List[str]

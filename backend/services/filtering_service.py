from domain.interfaces import FilteringService
from domain.entities import Movie
from typing import List, Dict, Any

def _matches(movie: Movie, filters: Dict[str, Any], watched_ids: set) -> bool:
    if movie.id in watched_ids:
        return False
    if filters.get("genres") and not set(filters["genres"]).intersection(set(movie.genres)):
        return False
    if filters.get("years") and movie.year not in filters["years"]:
        return False
    if filters.get("durations") and movie.duration > max(filters["durations"]):
        return False
    if filters.get("countries") and not set(filters["countries"]).intersection(set(movie.countries)):
        return False
    return True

class DefaultFilteringService(FilteringService):
    def filter(self, movies: List[Movie], filters: Dict[str, Any]) -> List[Movie]:
        watched_ids = set(filters.get("watched_ids", []))
        return [m for m in movies if _matches(m, filters, watched_ids)] 
import httpx
from domain.interfaces import TMDBService
from domain.entities import Movie
from core.settings import settings
from typing import List, Dict, Any
import redis
import hashlib
import json

class TMDBApiService(TMDBService):
    def __init__(self):
        self.api_key = settings.tmdb_api_key
        self.redis = redis.from_url(settings.redis_url)
        self.base_url = "https://api.themoviedb.org/3"

    def _cache_key(self, params: Dict[str, Any]) -> str:
        return f"tmdb:{hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}"

    def fetch_movies(self, filters: Dict[str, Any]) -> List[Movie]:
        cache_key = self._cache_key(filters)
        cached = self.redis.get(cache_key)
        if cached:
            return [Movie(**m) for m in json.loads(cached)]
        params = {"api_key": self.api_key, "language": "en-US", **filters}
        with httpx.Client() as client:
            resp = client.get(f"{self.base_url}/discover/movie", params=params)
            resp.raise_for_status()
            data = resp.json()
            movies = [Movie(
                id=str(m["id"]),
                title=m["title"],
                year=int(m["release_date"].split("-")[0]) if m.get("release_date") else 0,
                duration=m.get("runtime", 0),
                genres=[str(g) for g in m.get("genre_ids", [])],
                countries=[c["iso_3166_1"] for c in m.get("production_countries", [])] if m.get("production_countries") else [],
                description=m.get("overview", "")
            ) for m in data.get("results", [])]
            self.redis.set(cache_key, json.dumps([m.__dict__ for m in movies]), ex=60*60*24*7)
            return movies 
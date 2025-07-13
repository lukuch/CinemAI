import asyncio
from typing import Any, Dict, List

import httpx
import orjson
import redis

from core.settings import settings
from domain.entities import Movie
from domain.interfaces import TMDBService


class TMDBApiService(TMDBService):
    def __init__(self):
        self.api_key = settings.tmdb_api_key
        self.redis = redis.from_url(settings.redis_url)
        self.base_url = "https://api.themoviedb.org/3"

    def _get_genre_map(self):
        cache_key = "tmdb:genre_map"
        cached = self.redis.get(cache_key)
        if cached:
            return orjson.loads(cached)
        url = f"{self.base_url}/genre/movie/list"
        params = {"api_key": self.api_key, "language": "en-US"}
        resp = httpx.get(url, params=params)
        resp.raise_for_status()
        genres = resp.json()["genres"]
        genre_map = {g["id"]: g["name"] for g in genres}
        self.redis.set(cache_key, orjson.dumps(genre_map), ex=60 * 60 * 24 * 30)
        return genre_map

    async def fetch_movies(self, filters: Dict[str, Any]) -> List[Movie]:
        cached = self.redis.get("movies:all")
        if not cached:
            await self._fetch_and_cache_top_movies()
            cached = self.redis.get("movies:all")
        movies = orjson.loads(cached)

        filtered = [Movie(**m) for m in movies if self._matches(m, filters)]
        return filtered

    def _matches(self, movie: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        if "genres" in filters and filters["genres"]:
            if not any(g in movie["genres"] for g in filters["genres"]):
                return False
        if "year" in filters and filters["year"]:
            if movie["year"] != filters["year"]:
                return False
        if "countries" in filters and filters["countries"]:
            if not any(c in movie["countries"] for c in filters["countries"]):
                return False
        if "duration" in filters and filters["duration"]:
            if movie["duration"] != filters["duration"]:
                return False
        return True

    async def _fetch_and_cache_top_movies(self, total_movies: int = 10000, batch_size: int = 50) -> int:
        movies_per_page = 20
        total_pages = (total_movies + movies_per_page - 1) // movies_per_page
        genre_map = self._get_genre_map()
        all_movies_raw = []
        async with httpx.AsyncClient() as client:
            for page_start in range(1, total_pages + 1, batch_size):
                page_tasks = [
                    self._fetch_page(client, page) for page in range(page_start, min(page_start + batch_size, total_pages + 1))
                ]
                page_results = await asyncio.gather(*page_tasks)
                for page in page_results:
                    all_movies_raw.extend(page)
                await asyncio.sleep(1)
            all_movies_raw = all_movies_raw[:total_movies]
            movies = []
            for i in range(0, len(all_movies_raw), batch_size):
                batch = all_movies_raw[i : i + batch_size]
                detail_tasks = [self._fetch_details(client, m, genre_map) for m in batch]
                movies.extend([m for m in await asyncio.gather(*detail_tasks) if m is not None])
                await asyncio.sleep(1)
        self.redis.set("movies:all", orjson.dumps(movies))
        return len(movies)

    async def _fetch_page(self, client, page):
        params = {"api_key": self.api_key, "language": "en-US", "sort_by": "popularity.desc", "page": page}
        resp = await client.get(f"{self.base_url}/discover/movie", params=params)
        resp.raise_for_status()
        return resp.json()["results"]

    async def _fetch_details(self, client, m, genre_map):
        try:
            details_resp = await client.get(
                f"{self.base_url}/movie/{m['id']}", params={"api_key": self.api_key, "language": "en-US"}
            )
            if details_resp.status_code == 404:
                return None  # Movie not found, skip it
            details_resp.raise_for_status()
            details = details_resp.json()
            # Always use genre_ids from TMDB and map to names for 'genres'
            genre_ids = m.get("genre_ids", [])
            genres = [genre_map.get(str(gid), str(gid)) for gid in genre_ids]
            return {
                "id": str(m["id"]),
                "title": m["title"],
                "year": int(m["release_date"].split("-")[0]) if m.get("release_date") else 0,
                "duration": details.get("runtime", 0),
                "genres": genres,
                "countries": (
                    [c["iso_3166_1"] for c in details.get("production_countries", [])]
                    if details.get("production_countries")
                    else []
                ),
                "description": m.get("overview", ""),
            }
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None  # Movie not found, skip it
            raise  # Re-raise other errors

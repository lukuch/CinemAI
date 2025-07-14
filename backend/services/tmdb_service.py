import asyncio
from typing import Any, Dict, List, Optional

import httpx
import orjson
import redis
from injector import inject
from structlog.stdlib import BoundLogger

from core.settings import settings
from domain.entities import Movie
from domain.interfaces import TMDBService


class TMDBApiService(TMDBService):
    @inject
    def __init__(self, logger: BoundLogger):
        self.api_key = settings.tmdb_api_key
        self.redis = redis.from_url(settings.redis_url)
        self.base_url = "https://api.themoviedb.org/3"
        self.logger = logger

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

    async def enrich_movies_batch(self, movies_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich multiple movies in batch for better performance.
        """
        try:
            self.logger.info("Starting batch enrichment", total_movies=len(movies_data))

            # Filter movies that need enrichment (missing description)
            movies_to_enrich = []
            enriched_indices = []

            for i, movie_data in enumerate(movies_data):
                description = movie_data.get("description")
                title = movie_data.get("title")

                if (not description or not description.strip()) and title:
                    movies_to_enrich.append((i, movie_data))
                else:
                    enriched_indices.append(i)

            self.logger.info(
                "Enrichment analysis completed", movies_to_enrich=len(movies_to_enrich), already_enriched=len(enriched_indices)
            )

            if not movies_to_enrich:
                return movies_data  # No enrichment needed

            # Batch fetch descriptions from TMDB
            enriched_results = await self._fetch_descriptions_batch(movies_to_enrich)

            # Apply enrichment results
            result = movies_data.copy()
            successful_enrichments = 0
            for (original_index, movie_data), enriched_data in zip(movies_to_enrich, enriched_results):
                if enriched_data:
                    result[original_index] = enriched_data
                    successful_enrichments += 1

            self.logger.info(
                "Batch enrichment completed",
                successful_enrichments=successful_enrichments,
                enrichment_rate=successful_enrichments / len(movies_to_enrich) if movies_to_enrich else 0,
            )

            return result

        except Exception as e:
            self.logger.error("Batch enrichment failed", error=str(e))
            return movies_data  # Return original data on any error

    async def _fetch_descriptions_batch(self, movies_to_enrich: List[tuple]) -> List[Dict[str, Any]]:
        """
        Fetch descriptions for multiple movies in parallel.
        """
        try:
            # Create search tasks for all movies that need enrichment
            search_tasks = []
            for _, movie_data in movies_to_enrich:
                title = movie_data.get("title")
                year = movie_data.get("year")
                task = self._fetch_description_from_tmdb(title, year)
                search_tasks.append(task)

            # Execute all searches in parallel
            descriptions = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Apply descriptions to movies
            enriched_movies = []
            for (_, movie_data), description in zip(movies_to_enrich, descriptions):
                if isinstance(description, Exception) or not description:
                    enriched_movies.append(movie_data)  # Keep original if enrichment failed
                else:
                    enriched_data = movie_data.copy()
                    enriched_data["description"] = description
                    enriched_movies.append(enriched_data)

            return enriched_movies

        except Exception:
            # Return original movies if batch enrichment fails
            return [movie_data for _, movie_data in movies_to_enrich]

    async def _fetch_description_from_tmdb(self, title: str, year: Optional[int] = None) -> Optional[str]:
        """
        Fetch movie description from TMDB API.
        """
        try:
            # Use TMDB search API
            search_params = {"api_key": self.api_key, "language": "en-US", "query": title, "include_adult": False}

            if year:
                search_params["year"] = year

            url = f"{self.base_url}/search/movie"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=search_params)
                response.raise_for_status()

                results = response.json().get("results", [])

                if not results:
                    return None

                # Get the best match (first result)
                best_match = results[0]
                description = best_match.get("overview")

                if description and description.strip():
                    return description.strip()

                return None

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limit
                return None
            elif e.response.status_code == 404:
                return None
            else:
                return None
        except Exception:
            return None

    async def fetch_movies(self, filters: Dict[str, Any]) -> List[Movie]:
        self.logger.info("Fetching movies from TMDB", filters_applied=list(filters.keys()) if filters else "none")

        cached = self.redis.get("movies:all")
        if not cached:
            self.logger.info("Cache miss - fetching and caching top movies")
            await self._fetch_and_cache_top_movies()
            cached = self.redis.get("movies:all")
        else:
            self.logger.info("Cache hit - using cached movies")

        movies = orjson.loads(cached)
        self.logger.info("Movies loaded from cache", total_movies=len(movies))

        filtered = [Movie(**m) for m in movies if self._matches(m, filters)]
        self.logger.info(
            "Movies filtered",
            original_count=len(movies),
            filtered_count=len(filtered),
            filter_effectiveness=len(filtered) / len(movies) if movies else 0,
        )

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

    def _deduplicate_movies_by_id(self, movies: list) -> list:
        """Deduplicate a list of movie dicts by their 'id' field."""
        unique_movies_dict = {}
        for movie in movies:
            movie_id = movie.get("id")
            if movie_id and movie_id not in unique_movies_dict:
                unique_movies_dict[movie_id] = movie
        return list(unique_movies_dict.values())

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
        # Deduplicate by movie id
        unique_movies = self._deduplicate_movies_by_id(movies)
        duplicates_removed = len(movies) - len(unique_movies)
        # Ensure exactly 10,000 unique movies
        if len(unique_movies) > total_movies:
            unique_movies = unique_movies[:total_movies]
        self.logger.info(
            "Deduplication completed while fetching movies from TMDB and before caching",
            original_count=len(movies),
            unique_count=len(unique_movies),
            duplicates_removed=duplicates_removed,
        )
        self.redis.set("movies:all", orjson.dumps(unique_movies))
        return len(unique_movies)

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

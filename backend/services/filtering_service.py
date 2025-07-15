import unicodedata
from typing import Any, Dict, List

from injector import inject
from rapidfuzz import fuzz, process
from structlog.stdlib import BoundLogger

from domain.entities import Movie
from domain.interfaces import IFilteringService


def normalize_title(title):
    title = title.lower()
    title = "".join(
        c
        for c in unicodedata.normalize("NFD", title)
        if unicodedata.category(c) != "Mn"
    )
    return "".join(e for e in title if e.isalnum())


def build_watched_title_map(watched_movies):
    return {normalize_title(w.title): w for w in watched_movies}


def bulk_fuzzy_filter(candidates, watched_movies, threshold=85, year_tolerance=1):
    watched_titles = [normalize_title(w.title) for w in watched_movies]
    watched_years = [getattr(w, "year", None) for w in watched_movies]
    candidate_titles = [normalize_title(c.title) for c in candidates]
    candidate_years = [getattr(c, "year", None) for c in candidates]
    scores = process.cdist(candidate_titles, watched_titles, scorer=fuzz.ratio)
    filtered = []
    for i, candidate in enumerate(candidates):
        found = False
        for j, watched in enumerate(watched_movies):
            if scores[i][j] >= threshold:
                cy, wy = candidate_years[i], watched_years[j]
                if cy and wy and abs(int(cy) - int(wy)) <= year_tolerance:
                    found = True
                    break
        if not found:
            filtered.append(candidate)
    return filtered


class FilteringService(IFilteringService):
    @inject
    def __init__(self, logger: BoundLogger):
        self.logger = logger

    def filter(self, movies: List[Movie], filters: Dict[str, Any]) -> List[Movie]:
        self.logger.info(
            "Starting movie filtering",
            total_movies=len(movies),
            filters_applied=list(filters.keys()),
        )
        watched_movies = filters.get("watched_movies", [])
        fast_filtered = self._fast_path_filter(movies, watched_movies)
        self.logger.info(
            "Fast path filtering completed",
            filtered_count=len(fast_filtered),
            removed_count=len(movies) - len(fast_filtered),
        )

        filtered = bulk_fuzzy_filter(fast_filtered, watched_movies)
        self.logger.info(
            "Fuzzy filtering completed",
            filtered_count=len(filtered),
            removed_count=len(fast_filtered) - len(filtered),
        )

        result = self._apply_other_filters(filtered, filters)
        self.logger.info(
            "Additional filtering completed",
            filtered_count=len(result),
            removed_count=len(filtered) - len(result),
        )

        final_result = self._deduplicate(result)
        self.logger.info(
            "Deduplication completed",
            filtered_count=len(final_result),
            removed_count=len(result) - len(final_result),
        )
        self.logger.info(
            "Filtering process completed",
            final_count=len(final_result),
            total_removed=len(movies) - len(final_result),
            filter_effectiveness=(
                round(len(final_result) / len(movies), 3) if movies else 0
            ),
        )
        return final_result

    def _fast_path_filter(
        self, movies: List[Movie], watched_movies: List[Movie]
    ) -> List[Movie]:
        watched_title_map = build_watched_title_map(watched_movies)
        fast_filtered = []
        for m in movies:
            norm_title = normalize_title(m.title)
            watched = watched_title_map.get(norm_title)
            if watched:
                year1 = getattr(m, "year", None)
                year2 = getattr(watched, "year", None)
                if year1 and year2 and abs(int(year1) - int(year2)) <= 1:
                    continue
            fast_filtered.append(m)
        return fast_filtered

    def _apply_other_filters(
        self, movies: List[Movie], filters: Dict[str, Any]
    ) -> List[Movie]:
        result = []
        for movie in movies:
            if filters.get("genres") and not set(filters["genres"]).intersection(
                set(movie.genres)
            ):
                continue
            if filters.get("years") and movie.year not in filters["years"]:
                continue
            if filters.get("durations") and movie.duration > max(filters["durations"]):
                continue
            if filters.get("countries") and not set(filters["countries"]).intersection(
                set(movie.countries)
            ):
                continue
            result.append(movie)
        return result

    def _deduplicate(self, movies: List[Movie]) -> List[Movie]:
        seen = dict()
        unique = []
        for m in movies:
            key = (normalize_title(m.title), getattr(m, "year", None))
            if key not in seen:
                seen[key] = m
                unique.append(m)
        return unique

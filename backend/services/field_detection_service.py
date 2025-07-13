import re
from typing import Any, Dict, List, Optional

import pycountry
from injector import inject
from structlog.stdlib import BoundLogger

from domain.interfaces import FieldDetectionService


class DefaultFieldDetectionService(FieldDetectionService):
    """Service for detecting and validating movie data fields from various formats."""

    @inject
    def __init__(self, logger: BoundLogger):
        self.field_configs = self._get_field_configs()
        self.logger = logger

    def convert_movie_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert movie data to standard format using pattern-based detection.
        Returns None if pattern matching fails.
        """
        try:
            converted = {}

            for field_name, config in self.field_configs.items():
                value = self._detect_field(data, config, field_name)

                if config.get("required", False) and value is None:
                    return None

                if value is not None:
                    converted[field_name] = value

            return converted

        except Exception:
            return None

    def validate_movie_data(self, data: Dict[str, Any]) -> bool:
        """Validate that movie data has required fields and correct types."""
        for field_name, config in self.field_configs.items():
            if config.get("required", False):
                if field_name not in data:
                    self.logger.warning("Missing required field", field=field_name)
                    return False

                value = data[field_name]
                expected_type = config["type"]

                if expected_type == "str" and not isinstance(value, str):
                    self.logger.warning("Field must be a string", field=field_name, value_type=type(value))
                    return False
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    self.logger.warning("Field must be a number", field=field_name, value_type=type(value))
                    return False
                elif expected_type == "int" and not isinstance(value, int):
                    self.logger.warning("Field must be an integer", field=field_name, value_type=type(value))
                    return False
                elif expected_type == "list" and not isinstance(value, list):
                    self.logger.warning("Field must be a list", field=field_name, value_type=type(value))
                    return False

        return True

    def normalize_title_value(self, title: str) -> str:
        # Remove language artifacts like ' (en)' or ' (fr)' at the end
        return re.sub(r"\s*\([a-z]{2}\)\s*$", "", title, flags=re.IGNORECASE).strip()

    def map_country_to_tmdb(self, country: str) -> str:
        country = country.strip()
        try:
            match = (
                pycountry.countries.get(alpha_2=country.upper())
                or pycountry.countries.get(alpha_3=country.upper())
                or pycountry.countries.get(name=country.title())
            )
            if match:
                return match.alpha_2
            # Try common names and official names
            for c in pycountry.countries:
                if country.lower() in [c.name.lower(), getattr(c, "official_name", "").lower()]:
                    return c.alpha_2
            # Try partial match (e.g., 'UK' for 'United Kingdom')
            for c in pycountry.countries:
                if country.lower() in c.name.lower():
                    return c.alpha_2
        except Exception:
            pass
        return country

    def _detect_field(self, data: Dict[str, Any], field_config: Dict[str, Any], field_name: str = None) -> Optional[Any]:
        """
        Generic field detection method.
        """
        patterns = field_config["patterns"]
        field_type = field_config["type"]

        for pattern in patterns:
            if pattern in data:
                value = data[pattern]
                if field_type == "str":
                    return self._detect_str_field(value, field_name)
                elif field_type == "float":
                    return self._detect_float_field(value)
                elif field_type == "int":
                    return self._detect_int_field(value, field_config)
                elif field_type == "list":
                    return self._detect_list_field(value, field_name)
        return field_config.get("default")

    def _detect_str_field(self, value: Any, field_name: str = None) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            if field_name == "title":
                value = self.normalize_title_value(value)
            return value.strip()
        return None

    def _detect_float_field(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _detect_int_field(self, value: Any, field_config: Dict[str, Any]) -> Optional[int]:
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            if field_config.get("is_year"):
                year_match = re.search(r"(\d{4})", value)
                if year_match:
                    return int(year_match.group(1))
            else:
                try:
                    return int(value)
                except ValueError:
                    return None
        return None

    def _detect_list_field(self, value: Any, field_name: str = None) -> Optional[list]:
        if isinstance(value, list):
            if field_name == "countries":
                return [self.map_country_to_tmdb(str(item)) for item in value if item]
            return [str(item) for item in value if item]
        elif isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            if field_name == "countries":
                return [self.map_country_to_tmdb(item) for item in items]
            return items
        return None

    def _get_field_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get field detection configurations."""
        return {
            "title": {
                "patterns": [
                    "originalTitle",  # Highest priority
                    "title",
                    "original_title",
                    "primaryTitle",
                    "movie_name",
                    "name",
                    "film_title",
                    "internationalTitle",
                ],
                "type": "str",
                "required": True,
            },
            "rating": {
                "patterns": ["rating", "rating_score", "vote_average", "averageRating", "user_rating", "score", "rate"],
                "type": "float",
                "required": True,
            },
            "year": {
                "patterns": ["year", "release_year", "startYear", "production_year", "release_date", "releaseYear"],
                "type": "int",
                "required": True,
                "is_year": True,
            },
            "duration": {
                "patterns": ["duration", "runtime", "runtimeMinutes", "film_length", "length"],
                "type": "int",
                "required": True,
            },
            "genres": {"patterns": ["genres", "genre_list", "category_tags", "genre"], "type": "list", "required": True},
            "countries": {
                "patterns": ["countries", "country_list", "production_countries", "origin_countries", "country"],
                "type": "list",
                "required": True,
                "default": [],
            },
            "description": {
                "patterns": ["description", "plot_summary", "overview", "synopsis", "plot", "summary"],
                "type": "str",
                "required": True,
            },
            "watched_at": {
                "patterns": ["watched_at", "viewed_date", "watch_timestamp", "date", "watch_date", "viewDate"],
                "type": "str",
                "required": False,
            },
        }

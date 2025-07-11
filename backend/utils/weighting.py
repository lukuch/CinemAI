import math
from datetime import datetime

def rating_weight(rating: float) -> float:
    return math.exp(rating - 5)

def recency_weight(watched_at: str, now: datetime) -> float:
    # Exponential decay based on date
    watched = datetime.fromisoformat(watched_at)
    days = (now - watched).days
    return math.exp(-days / 365) 
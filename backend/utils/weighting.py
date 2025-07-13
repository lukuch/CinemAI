from datetime import datetime


def rating_weight(rating: float) -> float:
    # Gentler favoring: 1 → 0.15, 10 → 1.0, exponent 1.5
    # Exact values:
    # 1:   0.15
    # 2:   0.16
    # 3:   0.19
    # 4:   0.24
    # 5:   0.32
    # 6:   0.43
    # 7:   0.58
    # 8:   0.75
    # 9:   0.91
    # 10:  1.0
    return 0.15 + 0.85 * ((rating - 1) / 9) ** 1.5


def recency_weight(watched_at: str, now: datetime) -> float:
    watched = datetime.fromisoformat(watched_at)
    year = watched.year
    if year >= 2020:
        return 1.0
    elif year >= 1990:
        # Linear: 1990 → 0.85, 2020 → 1.0
        return 0.85 + 0.15 * ((year - 1990) / 30)
    elif year >= 1975:
        # Linear: 1975 → 0.3, 1990 → 0.85
        return 0.3 + 0.55 * ((year - 1975) / 15)
    else:
        return 0.3

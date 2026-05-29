from __future__ import annotations

from .data import Draw


def count_consecutive_pairs(reds: tuple[int, ...]) -> int:
    return sum(1 for a, b in zip(reds, reds[1:]) if b == a + 1)


def ac_value(reds: tuple[int, ...]) -> int:
    """AC 值：不同两两差值个数减去 (红球数 - 1)。"""
    diffs = {abs(b - a) for i, a in enumerate(reds) for b in reds[i + 1 :]}
    return max(0, len(diffs) - (len(reds) - 1))


def compute_features(draw: Draw, previous: Draw | None = None) -> dict[str, int | float]:
    reds = tuple(sorted(draw.red_balls))
    features: dict[str, int | float] = {
        "sum": sum(reds),
        "span": max(reds) - min(reds),
        "odd_count": sum(n % 2 for n in reds),
        "even_count": sum(1 for n in reds if n % 2 == 0),
        "small_count": sum(1 for n in reds if n <= 16),
        "large_count": sum(1 for n in reds if n >= 17),
        "zone_1_11": sum(1 for n in reds if 1 <= n <= 11),
        "zone_12_22": sum(1 for n in reds if 12 <= n <= 22),
        "zone_23_33": sum(1 for n in reds if 23 <= n <= 33),
        "consecutive_pairs": count_consecutive_pairs(reds),
        "ac_value": ac_value(reds),
        "tail_distinct": len({n % 10 for n in reds}),
        "prime_count": sum(1 for n in reds if n in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}),
    }
    features["repeat_count"] = len(set(reds) & set(previous.red_balls)) if previous else 0
    return features


def feature_table(draws: list[Draw]) -> list[dict[str, int | float | str]]:
    rows = []
    prev = None
    for draw in draws:
        row: dict[str, int | float | str] = {"issue": draw.issue, "date": draw.date, "blue_ball": draw.blue_ball}
        row.update(compute_features(draw, prev))
        rows.append(row)
        prev = draw
    return rows

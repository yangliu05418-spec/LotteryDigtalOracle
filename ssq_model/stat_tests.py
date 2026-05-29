from __future__ import annotations

import math
from collections import Counter
from statistics import NormalDist

from .data import Draw
from .theory import BLUE_BALLS, RED_BALLS


def _chi_square_sf_wilson_hilferty(x: float, k: int) -> float:
    """无 scipy 环境下的卡方右尾近似；用于探索性随机性检验。"""
    if k <= 0:
        return 1.0
    if x <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return max(0.0, min(1.0, 1 - NormalDist().cdf(z)))


def chi_square_uniform(counts: dict[int, int], categories: list[int] | tuple[int, ...], expected_total: int) -> dict[str, float]:
    expected = expected_total / len(categories)
    statistic = sum(((counts.get(c, 0) - expected) ** 2) / expected for c in categories)
    df = len(categories) - 1
    return {"statistic": statistic, "df": float(df), "p_value_approx": _chi_square_sf_wilson_hilferty(statistic, df)}


def ball_frequency_tests(draws: list[Draw]) -> dict[str, dict[str, float]]:
    red_counts = Counter(n for draw in draws for n in draw.red_balls)
    blue_counts = Counter(draw.blue_ball for draw in draws)
    return {
        "red_chi_square": chi_square_uniform(dict(red_counts), RED_BALLS, len(draws) * 6),
        "blue_chi_square": chi_square_uniform(dict(blue_counts), BLUE_BALLS, len(draws)),
    }


def runs_test_binary(values: list[int]) -> dict[str, float]:
    """二元序列游程检验。values 应只包含 0/1。"""
    if not values or any(v not in (0, 1) for v in values):
        raise ValueError("runs_test_binary 需要非空 0/1 序列")
    n1 = sum(values)
    n0 = len(values) - n1
    if n0 == 0 or n1 == 0:
        return {"runs": 1.0, "z": 0.0, "p_value_approx": 1.0}
    runs = 1 + sum(1 for a, b in zip(values, values[1:]) if a != b)
    mean = 1 + 2 * n0 * n1 / (n0 + n1)
    var = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / (((n0 + n1) ** 2) * (n0 + n1 - 1))
    z = (runs - mean) / math.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - NormalDist().cdf(abs(z)))
    return {"runs": float(runs), "z": z, "p_value_approx": max(0.0, min(1.0, p))}


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1], reverse=True)
    adjusted = [0.0] * m
    running = 1.0
    for rank_from_end, (idx, p) in enumerate(indexed, start=1):
        rank = m - rank_from_end + 1
        running = min(running, p * m / rank)
        adjusted[idx] = max(0.0, min(1.0, running))
    return adjusted

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random

from .models.base import Prediction
from .theory import BLUE_BALLS, RED_BALLS


@dataclass(frozen=True, order=True)
class Candidate:
    score: float
    red_balls: tuple[int, int, int, int, int, int]
    blue_ball: int


def _weighted_sample_without_replacement(items: list[int], weights: dict[int, float], k: int, rng: Random) -> tuple[int, ...]:
    remaining = list(items)
    chosen: list[int] = []
    for _ in range(k):
        total = sum(max(weights.get(item, 0.0), 0.0) for item in remaining)
        if total <= 0:
            pick = rng.choice(remaining)
        else:
            threshold = rng.random() * total
            acc = 0.0
            pick = remaining[-1]
            for item in remaining:
                acc += max(weights.get(item, 0.0), 0.0)
                if acc >= threshold:
                    pick = item
                    break
        chosen.append(pick)
        remaining.remove(pick)
    return tuple(sorted(chosen))


def _weighted_choice(items: list[int], weights: dict[int, float], rng: Random) -> int:
    total = sum(max(weights.get(item, 0.0), 0.0) for item in items)
    if total <= 0:
        return rng.choice(items)
    threshold = rng.random() * total
    acc = 0.0
    for item in items:
        acc += max(weights.get(item, 0.0), 0.0)
        if acc >= threshold:
            return item
    return items[-1]


def score_combination(prediction: Prediction, reds: tuple[int, ...], blue: int) -> float:
    eps = 1e-15
    return math.exp(sum(math.log(max(prediction.red_probs[n], eps)) for n in reds) + math.log(max(prediction.blue_probs[blue], eps)))


def generate_candidates(prediction: Prediction, *, top_k: int = 20, seed: int | None = 20260529, pool_size: int = 2000) -> list[Candidate]:
    """按模型概率采样候选组合，再按概率分数排序去重。

    返回仅是概率模型候选集，不能解释为投注建议。
    """
    if top_k <= 0:
        return []
    rng = Random(seed)
    seen: dict[tuple[tuple[int, ...], int], Candidate] = {}
    attempts = max(pool_size, top_k * 10)
    for _ in range(attempts):
        reds = _weighted_sample_without_replacement(list(RED_BALLS), prediction.red_probs, 6, rng)
        blue = _weighted_choice(list(BLUE_BALLS), prediction.blue_probs, rng)
        key = (reds, blue)
        if key not in seen:
            seen[key] = Candidate(score_combination(prediction, reds, blue), reds, blue)  # type: ignore[arg-type]
    return sorted(seen.values(), reverse=True)[:top_k]

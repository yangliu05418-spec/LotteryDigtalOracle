from __future__ import annotations

from dataclasses import dataclass
from math import comb
from random import Random

RED_BALLS = tuple(range(1, 34))
BLUE_BALLS = tuple(range(1, 17))
RED_PICK_COUNT = 6
TOTAL_RED_COMBINATIONS = comb(33, 6)
TOTAL_COMBINATIONS = TOTAL_RED_COMBINATIONS * 16


@dataclass(frozen=True, order=True)
class Combination:
    red_balls: tuple[int, int, int, int, int, int]
    blue_ball: int


def red_marginal_uniform() -> dict[int, float]:
    """理论随机模型下，每个红球出现在一期中的边际概率。"""
    p = RED_PICK_COUNT / len(RED_BALLS)
    return {n: p for n in RED_BALLS}


def blue_uniform() -> dict[int, float]:
    return {n: 1 / len(BLUE_BALLS) for n in BLUE_BALLS}


def random_combination(seed: int | None = None) -> Combination:
    rng = Random(seed)
    reds = tuple(sorted(rng.sample(list(RED_BALLS), RED_PICK_COUNT)))
    blue = rng.choice(list(BLUE_BALLS))
    return Combination(reds, blue)  # type: ignore[arg-type]


def combination_probability() -> float:
    return 1 / TOTAL_COMBINATIONS

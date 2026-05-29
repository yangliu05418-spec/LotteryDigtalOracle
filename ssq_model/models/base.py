from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    red_probs: dict[int, float]
    blue_probs: dict[int, float]
    model_name: str

    def validate(self) -> None:
        if set(self.red_probs) != set(range(1, 34)):
            raise ValueError("红球概率必须覆盖 1..33")
        if set(self.blue_probs) != set(range(1, 17)):
            raise ValueError("蓝球概率必须覆盖 1..16")
        if any(p < 0 or p > 1 for p in self.red_probs.values()):
            raise ValueError("红球边际概率必须在 0..1")
        if any(p < 0 for p in self.blue_probs.values()):
            raise ValueError("蓝球概率不能为负")
        if abs(sum(self.blue_probs.values()) - 1.0) > 1e-8:
            raise ValueError("蓝球概率和必须为 1")


def normalize_blue(weights: dict[int, float]) -> dict[int, float]:
    total = sum(max(0.0, w) for w in weights.values())
    if total <= 0:
        return {i: 1 / 16 for i in range(1, 17)}
    return {i: max(0.0, weights.get(i, 0.0)) / total for i in range(1, 17)}


def scale_red_to_six(weights: dict[int, float]) -> dict[int, float]:
    positives = {i: max(0.0, weights.get(i, 0.0)) for i in range(1, 34)}
    total = sum(positives.values())
    if total <= 0:
        return {i: 6 / 33 for i in range(1, 34)}
    return {i: min(1.0, positives[i] * 6 / total) for i in range(1, 34)}

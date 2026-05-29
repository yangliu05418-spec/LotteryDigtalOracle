from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Callable

from .data import Draw
from .models.base import Prediction

EPS = 1e-12


@dataclass(frozen=True)
class BacktestResult:
    rows: list[dict[str, float | int | str]]

    @property
    def n_predictions(self) -> int:
        return len(self.rows)

    def summary(self) -> dict[str, float | int]:
        if not self.rows:
            return {"n_predictions": 0}
        keys = ["red_log_loss", "blue_log_loss", "red_brier", "blue_brier"]
        return {"n_predictions": len(self.rows), **{k: mean(float(r[k]) for r in self.rows) for k in keys}}


def red_log_loss(prediction: Prediction, draw: Draw, *, eps: float = EPS) -> float:
    # 多标签边际 log loss：只对实际出现的 6 个红球取平均。概率为 0 时裁剪。
    return -mean(math.log(max(prediction.red_probs[n], eps)) for n in draw.red_balls)


def blue_log_loss(prediction: Prediction, draw: Draw, *, eps: float = EPS) -> float:
    return -math.log(max(prediction.blue_probs[draw.blue_ball], eps))


def red_brier(prediction: Prediction, draw: Draw) -> float:
    actual = set(draw.red_balls)
    return mean((prediction.red_probs[i] - (1.0 if i in actual else 0.0)) ** 2 for i in range(1, 34))


def blue_brier(prediction: Prediction, draw: Draw) -> float:
    return mean((prediction.blue_probs[i] - (1.0 if i == draw.blue_ball else 0.0)) ** 2 for i in range(1, 17))


def rolling_backtest(draws: list[Draw], model_factory: Callable[[], object], *, min_train_size: int = 200) -> BacktestResult:
    """rolling-origin 回测：第 t 期只能用 t 之前的数据训练。"""
    if min_train_size < 1:
        raise ValueError("min_train_size 必须为正数")
    rows: list[dict[str, float | int | str]] = []
    ordered = sorted(draws, key=lambda d: int(d.issue))
    for idx in range(min_train_size, len(ordered)):
        train = ordered[:idx]
        target = ordered[idx]
        model = model_factory()
        prediction = model.fit(train).predict_proba()  # type: ignore[attr-defined]
        rows.append(
            {
                "issue": target.issue,
                "train_size": len(train),
                "red_log_loss": red_log_loss(prediction, target),
                "blue_log_loss": blue_log_loss(prediction, target),
                "red_brier": red_brier(prediction, target),
                "blue_brier": blue_brier(prediction, target),
            }
        )
    return BacktestResult(rows)

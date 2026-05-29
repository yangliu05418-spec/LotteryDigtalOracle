from __future__ import annotations

from collections import Counter

from ..data import Draw
from .base import Prediction, normalize_blue, scale_red_to_six


class FrequencyModel:
    """历史边际频率模型。红球概率表示“某球在一期中出现”的边际概率。"""

    name = "frequency"

    def __init__(self) -> None:
        self._draws: list[Draw] = []

    def fit(self, draws: list[Draw]) -> "FrequencyModel":
        if not draws:
            raise ValueError("FrequencyModel 至少需要 1 期训练数据")
        self._draws = list(draws)
        return self

    def predict_proba(self) -> Prediction:
        n = len(self._draws)
        red_counts = Counter(ball for draw in self._draws for ball in draw.red_balls)
        blue_counts = Counter(draw.blue_ball for draw in self._draws)
        red = {i: red_counts.get(i, 0) / n for i in range(1, 34)}
        blue = normalize_blue({i: blue_counts.get(i, 0) for i in range(1, 17)})
        pred = Prediction(red, blue, self.name)
        pred.validate()
        return pred


class RollingWindowModel(FrequencyModel):
    name = "rolling_window"

    def __init__(self, window: int = 200) -> None:
        super().__init__()
        if window <= 0:
            raise ValueError("window 必须为正数")
        self.window = window

    def fit(self, draws: list[Draw]) -> "RollingWindowModel":
        return super().fit(list(draws)[-self.window:])  # type: ignore[return-value]


class ExponentialDecayModel(FrequencyModel):
    name = "exponential_decay"

    def __init__(self, decay: float = 0.995) -> None:
        super().__init__()
        if not 0 < decay <= 1:
            raise ValueError("decay 必须在 (0, 1]")
        self.decay = decay

    def predict_proba(self) -> Prediction:
        weights = [self.decay ** (len(self._draws) - idx - 1) for idx in range(len(self._draws))]
        total_draw_weight = sum(weights)
        red_weights = {i: 0.0 for i in range(1, 34)}
        blue_weights = {i: 0.0 for i in range(1, 17)}
        for draw, weight in zip(self._draws, weights):
            for ball in draw.red_balls:
                red_weights[ball] += weight
            blue_weights[draw.blue_ball] += weight
        red = {i: red_weights[i] / total_draw_weight for i in range(1, 34)}
        blue = normalize_blue(blue_weights)
        pred = Prediction(red, blue, self.name)
        pred.validate()
        return pred

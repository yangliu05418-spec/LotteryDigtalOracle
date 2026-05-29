from __future__ import annotations

from collections import Counter

from ..data import Draw
from .base import Prediction, normalize_blue, scale_red_to_six


class BayesianSmoothingModel:
    """轻量贝叶斯/经验贝叶斯平滑：红球 Beta-Binomial 边际，蓝球 Dirichlet-Multinomial。"""

    name = "bayesian_smoothing"

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha 必须为正数")
        self.alpha = alpha
        self._draws: list[Draw] = []

    def fit(self, draws: list[Draw]) -> "BayesianSmoothingModel":
        if not draws:
            raise ValueError("BayesianSmoothingModel 至少需要 1 期训练数据")
        self._draws = list(draws)
        return self

    def predict_proba(self) -> Prediction:
        n = len(self._draws)
        red_counts = Counter(ball for draw in self._draws for ball in draw.red_balls)
        blue_counts = Counter(draw.blue_ball for draw in self._draws)
        # 让红球边际概率和严格为 6：先对“槽位频数”做 Dirichlet 平滑，再缩放到 6 个包含概率。
        red_slot_weights = {i: red_counts.get(i, 0) + self.alpha for i in range(1, 34)}
        red = scale_red_to_six(red_slot_weights)
        blue = normalize_blue({i: blue_counts.get(i, 0) + self.alpha for i in range(1, 17)})
        pred = Prediction(red, blue, self.name)
        pred.validate()
        return pred

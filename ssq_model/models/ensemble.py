from __future__ import annotations

from ..data import Draw
from .base import Prediction, normalize_blue, scale_red_to_six


class EnsembleModel:
    """对多个概率模型做线性集成。"""

    name = "ensemble"

    def __init__(self, models: list, weights: list[float] | None = None) -> None:
        if not models:
            raise ValueError("EnsembleModel 至少需要一个子模型")
        self.models = models
        self.weights = weights or [1.0] * len(models)
        if len(self.weights) != len(self.models):
            raise ValueError("weights 长度必须等于 models 长度")
        if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError("weights 必须非负且和大于 0")

    def fit(self, draws: list[Draw]) -> "EnsembleModel":
        for model in self.models:
            model.fit(draws)
        return self

    def predict_proba(self) -> Prediction:
        total = sum(self.weights)
        red = {i: 0.0 for i in range(1, 34)}
        blue = {i: 0.0 for i in range(1, 17)}
        for model, weight in zip(self.models, self.weights):
            pred = model.predict_proba()
            for i, p in pred.red_probs.items():
                red[i] += weight * p / total
            for i, p in pred.blue_probs.items():
                blue[i] += weight * p / total
        pred = Prediction(scale_red_to_six(red), normalize_blue(blue), self.name)
        pred.validate()
        return pred

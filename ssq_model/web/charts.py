from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from ..data import Draw


@dataclass(frozen=True)
class ChartSpec:
    title: str
    x: list[str]
    y: list[float]
    x_label: str
    y_label: str


def build_frequency_chart_spec(values: Mapping[int, float], *, title: str, x_label: str, y_label: str) -> ChartSpec:
    items = sorted(values.items())
    return ChartSpec(
        title=title,
        x=[str(k) for k, _ in items],
        y=[float(v) for _, v in items],
        x_label=x_label,
        y_label=y_label,
    )


def red_frequency(draws: list[Draw]) -> dict[int, float]:
    counter = Counter(ball for draw in draws for ball in draw.red_balls)
    denominator = max(1, len(draws))
    return {i: counter.get(i, 0) / denominator for i in range(1, 34)}


def blue_frequency(draws: list[Draw]) -> dict[int, float]:
    counter = Counter(draw.blue_ball for draw in draws)
    denominator = max(1, len(draws))
    return {i: counter.get(i, 0) / denominator for i in range(1, 17)}


def red_sum_distribution(draws: list[Draw]) -> dict[int, int]:
    return dict(sorted(Counter(sum(draw.red_balls) for draw in draws).items()))


def span_distribution(draws: list[Draw]) -> dict[int, int]:
    return dict(sorted(Counter(max(draw.red_balls) - min(draw.red_balls) for draw in draws).items()))


def to_plotly_bar(spec: ChartSpec):
    try:
        import plotly.express as px  # type: ignore
    except ImportError:
        return spec
    return px.bar(x=spec.x, y=spec.y, labels={"x": spec.x_label, "y": spec.y_label}, title=spec.title)

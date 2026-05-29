from __future__ import annotations

from pathlib import Path
from typing import Any

from ..data import Draw
from ..generator import generate_candidates
from ..models import BayesianSmoothingModel, EnsembleModel, ExponentialDecayModel, FrequencyModel, PyMCJointBayesianModel
from ..reporting import write_csv_rows, write_markdown_report
from ..stat_tests import ball_frequency_tests
from ..theory import TOTAL_COMBINATIONS


def web_model_from_name(name: str):
    if name == "bayesian":
        return BayesianSmoothingModel(alpha=1.0)
    if name == "pymc":
        return PyMCJointBayesianModel(alpha=1.0, beta=1.0)
    if name == "ensemble":
        return EnsembleModel([FrequencyModel(), BayesianSmoothingModel(alpha=1.0), ExponentialDecayModel(decay=0.995)])
    raise ValueError(f"Web App 不支持的模型: {name}")


def generate_candidate_rows(draws: list[Draw], *, model_name: str, top_k: int, seed: int, pool_size: int) -> list[dict[str, Any]]:
    prediction = web_model_from_name(model_name).fit(draws).predict_proba()
    candidates = generate_candidates(prediction, top_k=top_k, seed=seed, pool_size=pool_size)
    return [
        {
            "序号": idx,
            "红球": " ".join(f"{n:02d}" for n in candidate.red_balls),
            "蓝球": f"{candidate.blue_ball:02d}",
            "score": f"{candidate.score:.8e}",
        }
        for idx, candidate in enumerate(candidates, start=1)
    ]


def pymc_summary_rows(draws: list[Draw], *, quick: bool = True, alpha: float = 1.0, beta: float = 1.0) -> dict[str, list[dict[str, Any]]]:
    sample = draws[-min(500, len(draws)) :] if quick else draws
    prediction = PyMCJointBayesianModel(alpha=alpha, beta=beta).fit(sample).predict_proba()
    red_top = sorted(prediction.red_probs.items(), key=lambda kv: kv[1], reverse=True)[:10]
    blue_top = sorted(prediction.blue_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "red": [{"红球": f"{ball:02d}", "后验均值": prob} for ball, prob in red_top],
        "blue": [{"蓝球": f"{ball:02d}", "后验均值": prob} for ball, prob in blue_top],
    }


def create_web_report(draws: list[Draw], output_dir: str | Path = "outputs") -> Path:
    out = Path(output_dir)
    tests = ball_frequency_tests(draws)
    report = write_markdown_report(
        out / "web_report.md",
        title="双色球本地 Web App 研究报告",
        sections=[
            ("边界声明", "仅用于概率建模、随机性检验和历史回测研究，不构成投注建议。"),
            ("数据概览", f"样本期数：{len(draws)}\n\n理论组合空间：{TOTAL_COMBINATIONS:,}"),
            ("随机性检验", "\n".join(f"- {k}: statistic={v['statistic']:.4f}, p≈{v['p_value_approx']:.6f}" for k, v in tests.items())),
        ],
    )
    write_csv_rows(out / "web_quality.csv", [{"样本期数": len(draws), "理论组合空间": TOTAL_COMBINATIONS}])
    return report

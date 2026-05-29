from __future__ import annotations

import platform
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ..data import Draw
from ..theory import TOTAL_COMBINATIONS

ACADEMIC_NOTICE = "仅用于概率建模、统计检验与历史回测研究；不构成投注建议，不承诺命中。"


def is_module_available(name: str) -> bool:
    return find_spec(name) is not None


def build_dashboard_summary(draws: list[Draw], data_path: str | Path) -> dict[str, Any]:
    latest = draws[-1] if draws else None
    return {
        "data_path": str(data_path),
        "sample_count": len(draws),
        "latest_issue": latest.issue if latest else "无数据",
        "latest_date": latest.date if latest else "无数据",
        "total_combinations": f"{TOTAL_COMBINATIONS:,}",
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "is_apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "pymc_available": is_module_available("pymc"),
        "jax_available": is_module_available("jax"),
        "streamlit_available": is_module_available("streamlit"),
        "academic_notice": ACADEMIC_NOTICE,
    }


def data_quality_rows(draws: list[Draw]) -> list[dict[str, str]]:
    checks = [
        ("样本数量", len(draws) > 0, str(len(draws))),
        ("期号唯一", len({d.issue for d in draws}) == len(draws), "无重复"),
        ("红球合法", all(len(d.red_balls) == 6 and len(set(d.red_balls)) == 6 and all(1 <= n <= 33 for n in d.red_balls) for d in draws), "1..33 且无重复"),
        ("蓝球合法", all(1 <= d.blue_ball <= 16 for d in draws), "1..16"),
    ]
    return [{"检查项": name, "状态": "通过" if ok else "失败", "说明": detail} for name, ok, detail in checks]

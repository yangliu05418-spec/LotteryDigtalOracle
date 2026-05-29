from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

DEFAULT_DATA_PATH = Path("数据") / "历史数据.csv"
REQUIRED_COLUMNS = ["期号", "日期", "红球-1", "红球-2", "红球-3", "红球-4", "红球-5", "红球-6", "蓝球"]


@dataclass(frozen=True, order=True)
class Draw:
    issue: str
    date: str
    red_balls: tuple[int, int, int, int, int, int]
    blue_ball: int

    @property
    def parsed_date(self) -> datetime:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(self.date, fmt)
            except ValueError:
                pass
        raise ValueError(f"日期不可解析: issue={self.issue}, date={self.date!r}")


def _row_to_draw(row: dict[str, str]) -> Draw:
    reds = tuple(sorted(int(row[f"红球-{i}"]) for i in range(1, 7)))
    return Draw(
        issue=str(row["期号"]).strip(),
        date=str(row["日期"]).strip(),
        red_balls=reds,  # type: ignore[arg-type]
        blue_ball=int(row["蓝球"]),
    )


def load_draws(path: str | Path = DEFAULT_DATA_PATH, *, validate: bool = True) -> list[Draw]:
    """读取 UTF-8/UTF-8-BOM CSV，并按期号升序返回开奖样本。"""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV 缺少表头")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV 缺少字段: {missing}")
        draws = [_row_to_draw(row) for row in reader]
    draws.sort(key=lambda d: int(d.issue))
    if validate:
        validate_draws(draws)
    return draws


def validate_draws(draws: Iterable[Draw]) -> None:
    """校验期号唯一、日期可解析、红球合法无重复、蓝球合法。"""
    seen: set[str] = set()
    for draw in draws:
        if not draw.issue:
            raise ValueError("存在空期号")
        if draw.issue in seen:
            raise ValueError(f"期号重复: {draw.issue}")
        seen.add(draw.issue)
        _ = draw.parsed_date
        if len(draw.red_balls) != 6:
            raise ValueError(f"红球数量不是 6: issue={draw.issue}")
        if len(set(draw.red_balls)) != 6:
            raise ValueError(f"红球重复: issue={draw.issue}, reds={draw.red_balls}")
        if any(n < 1 or n > 33 for n in draw.red_balls):
            raise ValueError(f"红球超出 1..33: issue={draw.issue}, reds={draw.red_balls}")
        if draw.blue_ball < 1 or draw.blue_ball > 16:
            raise ValueError(f"蓝球超出 1..16: issue={draw.issue}, blue={draw.blue_ball}")

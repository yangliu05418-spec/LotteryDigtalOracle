from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def write_markdown_report(path: str | Path, *, title: str, sections: Iterable[tuple[str, str]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "> 学术用途声明：本报告仅用于概率建模、随机基准比较与历史回测研究，不构成投注建议，不承诺命中。", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip(), ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def write_csv_rows(path: str | Path, rows: list[dict]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8-sig")
        return output
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output

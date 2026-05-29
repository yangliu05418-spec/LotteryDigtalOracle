"""长期可用的中彩网双色球历史开奖数据爬虫。

默认抓取从第一期到最新一期的全部双色球开奖数据，并写入 CSV：
期号, 日期, 红球-1..红球-6, 蓝球
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Iterable

import requests


API_URL = "https://jc.zhcw.com/port/client_json.php"
CSV_HEADER = ["期号", "日期", "红球-1", "红球-2", "红球-3", "红球-4", "红球-5", "红球-6", "蓝球"]
DEFAULT_OUTPUT = Path("数据") / "历史数据.csv"

HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Referer": "https://www.zhcw.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


def strip_jsonp(text: str) -> str:
    """去掉可选 JSONP 包装，返回纯 JSON 字符串。"""
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    match = re.match(r"^[\w$.]+\((.*)\);?\s*$", stripped, flags=re.S)
    if not match:
        raise ValueError("响应既不是 JSON，也不是可识别的 JSONP")
    return match.group(1).strip()


def parse_payload(text: str) -> dict:
    """解析接口返回的 JSON/JSONP 文本。"""
    return json.loads(strip_jsonp(text))


def normalize_record(raw: dict) -> list[str]:
    """将接口单条记录标准化为 CSV 行。"""
    issue = str(raw.get("issue", "")).strip()
    open_time = str(raw.get("openTime", "")).strip()
    red_balls = str(raw.get("frontWinningNum", "")).split()
    blue_ball = str(raw.get("backWinningNum", "")).strip()

    if not issue:
        raise ValueError(f"缺少期号: {raw!r}")
    if not open_time:
        raise ValueError(f"缺少开奖日期: {raw!r}")
    if len(red_balls) != 6:
        raise ValueError(f"红球数量不是 6 个: issue={issue}, frontWinningNum={raw.get('frontWinningNum')!r}")
    if not blue_ball:
        raise ValueError(f"缺少蓝球: issue={issue}")

    return [issue, open_time, *red_balls, blue_ball]


def sort_records_ascending(rows: Iterable[list[str]]) -> list[list[str]]:
    """按期号从第一期到最新一期升序排列。"""
    return sorted(rows, key=lambda row: int(row[0]))


def build_params(page_num: int, page_size: int) -> dict[str, str | int | float]:
    """构造请求参数；不传 callback 时接口直接返回 JSON。"""
    return {
        "transactionType": "10001001",
        "lotteryId": "1",
        "issueCount": "10000",
        "startIssue": "",
        "endIssue": "",
        "startDate": "",
        "endDate": "",
        "type": "0",
        "pageNum": page_num,
        "pageSize": page_size,
        "tt": random.random(),
    }


def fetch_page(
    session: requests.Session,
    page_num: int,
    page_size: int,
    *,
    timeout: float = 20,
    retries: int = 3,
) -> dict:
    """请求单页数据，带简单重试。"""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                API_URL,
                params=build_params(page_num, page_size),
                headers=HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            return parse_payload(response.text)
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"请求第 {page_num} 页失败，已重试 {retries} 次: {last_error}") from last_error


def crawl_all(
    *,
    page_size: int = 200,
    sleep_seconds: float = 0.15,
    timeout: float = 20,
    retries: int = 3,
    verbose: bool = True,
) -> list[list[str]]:
    """抓取全部双色球历史数据，并返回按期号升序排列的 CSV 行。"""
    rows: list[list[str]] = []
    page_num = 1
    total_pages: int | None = None

    with requests.Session() as session:
        while True:
            payload = fetch_page(session, page_num, page_size, timeout=timeout, retries=retries)
            data = payload.get("data") or []
            if not isinstance(data, list):
                raise ValueError(f"接口 data 字段不是列表: page={page_num}, data={data!r}")

            if total_pages is None:
                total_pages = int(payload.get("pages") or 0)
                if verbose:
                    total = payload.get("total", "未知")
                    print(f"接口返回总记录数: {total}，总页数: {total_pages}，每页: {page_size}")

            if not data:
                if verbose:
                    print(f"第 {page_num} 页为空，停止。")
                break

            for item in data:
                rows.append(normalize_record(item))

            if verbose:
                print(f"已抓取第 {page_num}/{total_pages or '?'} 页，累计 {len(rows)} 条")

            if total_pages and page_num >= total_pages:
                break

            page_num += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return sort_records_ascending(rows)


def write_csv(rows: Iterable[list[str]], output: Path = DEFAULT_OUTPUT) -> None:
    """写入 UTF-8 BOM CSV，方便 Excel 直接打开中文表头。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取中彩网双色球全部历史开奖数据")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="输出 CSV 路径")
    parser.add_argument("--page-size", type=int, default=200, help="每页数量，默认 200")
    parser.add_argument("--sleep", type=float, default=0.15, help="每页请求间隔秒数，默认 0.15")
    parser.add_argument("--timeout", type=float, default=20, help="单次请求超时秒数，默认 20")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数，默认 3")
    parser.add_argument("--quiet", action="store_true", help="减少输出")
    args = parser.parse_args()

    rows = crawl_all(
        page_size=args.page_size,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
        retries=args.retries,
        verbose=not args.quiet,
    )
    write_csv(rows, args.output)

    if rows:
        print(f"完成：共 {len(rows)} 条，范围 {rows[0][0]} 至 {rows[-1][0]}，文件：{args.output}")
    else:
        print(f"完成：未抓取到数据，文件：{args.output}")


if __name__ == "__main__":
    main()

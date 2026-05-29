from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from statistics import mean

from .backtest import rolling_backtest
from .data import DEFAULT_DATA_PATH, Draw, load_draws
from .features import feature_table
from .generator import generate_candidates
from .models import (
    BayesianSmoothingModel,
    EnsembleModel,
    ExponentialDecayModel,
    FrequencyModel,
    PyMCJointBayesianModel,
    PyMCSamplingConfig,
    RollingWindowModel,
)
from .reporting import write_csv_rows, write_markdown_report
from .stat_tests import ball_frequency_tests
from .theory import TOTAL_COMBINATIONS, blue_uniform, red_marginal_uniform

DISCLAIMER = "学术用途：仅研究概率分布、随机性检验和历史回测；不构成投注建议，不承诺命中。"


def _load(path: str | Path) -> list[Draw]:
    return load_draws(path)


def _model_from_name(name: str):
    if name == "frequency":
        return FrequencyModel()
    if name == "rolling":
        return RollingWindowModel(window=200)
    if name == "decay":
        return ExponentialDecayModel(decay=0.995)
    if name == "bayesian":
        return BayesianSmoothingModel(alpha=1.0)
    if name == "ensemble":
        return EnsembleModel([FrequencyModel(), BayesianSmoothingModel(alpha=1.0), ExponentialDecayModel(decay=0.995)])
    if name == "pymc":
        return PyMCJointBayesianModel(alpha=1.0, beta=1.0)
    raise ValueError(f"未知模型: {name}")


def mac_bootstrap_commands() -> list[str]:
    return [
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "python3 -m pip install --upgrade pip setuptools wheel",
        "python3 -m pip install -e '.[macos-arm64]'",
        "python3 -c \"import pymc, arviz, jax, numpyro, blackjax; print('PyMC Apple Silicon environment OK')\"",
        "python3 -m ssq_model analyze",
        "python3 -m ssq_model pymc-fit --quick --no-sample",
    ]


def streamlit_launch_command(*, port: int = 8501) -> list[str]:
    return ["python3", "-m", "streamlit", "run", "ssq_model/web_app.py", f"--server.port={port}"]


def cmd_analyze(args) -> None:
    draws = _load(args.data)
    sums = [sum(d.red_balls) for d in draws]
    print(DISCLAIMER)
    print(f"样本期数: {len(draws)}")
    print(f"期号范围: {draws[0].issue} .. {draws[-1].issue}")
    print(f"理论组合空间: C(33,6)×16 = {TOTAL_COMBINATIONS:,}")
    print(f"红球和值均值: {mean(sums):.3f}")
    print(f"理论红球边际概率示例 P(红球1出现): {red_marginal_uniform()[1]:.6f}")
    print(f"理论蓝球概率: {next(iter(blue_uniform().values())):.6f}")


def cmd_test_randomness(args) -> None:
    draws = _load(args.data)
    print(DISCLAIMER)
    for name, result in ball_frequency_tests(draws).items():
        print(f"{name}: statistic={result['statistic']:.4f}, df={result['df']:.0f}, p≈{result['p_value_approx']:.6f}")
    print("说明：无 scipy 时 p 值为 Wilson-Hilferty 正态近似，仅作探索性参考。")


def cmd_backtest(args) -> None:
    draws = _load(args.data)
    result = rolling_backtest(draws, lambda: _model_from_name(args.model), min_train_size=args.min_train_size)
    summary = result.summary()
    if args.output:
        write_csv_rows(args.output, result.rows)
    print(DISCLAIMER)
    print(f"模型: {args.model}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    if args.output:
        print(f"逐期回测已写入: {args.output}")


def cmd_train(args) -> None:
    draws = _load(args.data)
    model = _model_from_name(args.model).fit(draws)
    prediction = model.predict_proba()
    print(DISCLAIMER)
    print(f"已训练模型: {prediction.model_name}")
    print("红球概率 Top 10:", sorted(prediction.red_probs.items(), key=lambda kv: kv[1], reverse=True)[:10])
    print("蓝球概率 Top 5:", sorted(prediction.blue_probs.items(), key=lambda kv: kv[1], reverse=True)[:5])


def cmd_predict(args) -> None:
    draws = _load(args.data)
    prediction = _model_from_name(args.model).fit(draws).predict_proba()
    candidates = generate_candidates(prediction, top_k=args.top_k, seed=args.seed, pool_size=args.pool_size)
    print(DISCLAIMER)
    print(f"模型: {prediction.model_name}; 候选组合仅为概率模型输出。")
    for idx, cand in enumerate(candidates, start=1):
        reds = " ".join(f"{n:02d}" for n in cand.red_balls)
        print(f"{idx:02d}. 红球 {reds} | 蓝球 {cand.blue_ball:02d} | score={cand.score:.8e}")


def cmd_report(args) -> None:
    draws = _load(args.data)
    tests = ball_frequency_tests(draws)
    features = feature_table(draws)
    result = rolling_backtest(draws, lambda: _model_from_name(args.model), min_train_size=args.min_train_size)
    summary = result.summary()
    out_dir = Path(args.output_dir)
    write_csv_rows(out_dir / "features.csv", features)
    write_csv_rows(out_dir / "backtest.csv", result.rows)
    report = write_markdown_report(
        out_dir / "report.md",
        title="双色球概率建模研究报告",
        sections=[
            ("边界声明", DISCLAIMER),
            ("数据概览", f"样本期数：{len(draws)}\n\n期号范围：{draws[0].issue} .. {draws[-1].issue}\n\n理论组合空间：{TOTAL_COMBINATIONS:,}"),
            ("随机性检验", "\n".join(f"- {k}: statistic={v['statistic']:.4f}, df={v['df']:.0f}, p≈{v['p_value_approx']:.6f}" for k, v in tests.items())),
            ("回测摘要", "\n".join(f"- {k}: {v:.6f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in summary.items())),
            ("解释限制", "历史频率偏差不等于未来开奖规律；模型输出必须与理论随机基准共同解读。"),
        ],
    )
    print(DISCLAIMER)
    print(f"报告已生成: {report}")


def _sampling_config_from_args(args) -> PyMCSamplingConfig:
    return PyMCSamplingConfig(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        random_seed=args.seed,
        target_accept=args.target_accept,
        sampler=args.sampler,
    )


def cmd_pymc_fit(args) -> None:
    draws = _load(args.data)
    if args.quick:
        draws = draws[-min(len(draws), 500) :]
    model = PyMCJointBayesianModel(alpha=args.alpha, beta=args.beta).fit(draws)
    prediction = model.predict_proba()
    print(DISCLAIMER)
    print("PyMC 贝叶斯模型: joint red Beta-Binomial + blue Dirichlet")
    print(f"训练样本期数: {len(draws)}")
    print("红球后验均值 Top 10:", sorted(prediction.red_probs.items(), key=lambda kv: kv[1], reverse=True)[:10])
    print("蓝球后验均值 Top 5:", sorted(prediction.blue_probs.items(), key=lambda kv: kv[1], reverse=True)[:5])
    if args.no_sample:
        print("已跳过 MCMC；当前输出为共轭后验解析均值，适合快速验证部署。")
        return
    trace = model.sample(_sampling_config_from_args(args))
    print(f"MCMC 采样完成: {trace}")


def cmd_pymc_predict(args) -> None:
    draws = _load(args.data)
    if args.quick:
        draws = draws[-min(len(draws), 500) :]
    model = PyMCJointBayesianModel(alpha=args.alpha, beta=args.beta).fit(draws)
    if not args.no_sample:
        model.sample(_sampling_config_from_args(args))
    prediction = model.predict_proba()
    candidates = generate_candidates(prediction, top_k=args.top_k, seed=args.seed, pool_size=args.pool_size)
    print(DISCLAIMER)
    print("PyMC 候选组合仅为贝叶斯后验概率模型输出。")
    for idx, cand in enumerate(candidates, start=1):
        reds = " ".join(f"{n:02d}" for n in cand.red_balls)
        print(f"{idx:02d}. 红球 {reds} | 蓝球 {cand.blue_ball:02d} | score={cand.score:.8e}")


def cmd_mac_bootstrap(args) -> None:
    print("Apple Silicon macOS 部署命令（请在 Mac 终端运行）:")
    for command in mac_bootstrap_commands():
        print(command)


def cmd_web(args) -> None:
    command = streamlit_launch_command(port=args.server_port)
    if args.print_command:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m ssq_model", description="双色球概率建模研究 CLI")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="历史数据 CSV 路径")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("analyze", help="基础数据画像与理论基准").set_defaults(func=cmd_analyze)
    sub.add_parser("test-randomness", help="随机性/均匀性检验").set_defaults(func=cmd_test_randomness)

    p_back = sub.add_parser("backtest", help="rolling-origin 回测")
    p_back.add_argument("--model", choices=["frequency", "rolling", "decay", "bayesian", "ensemble", "pymc"], default="bayesian")
    p_back.add_argument("--min-train-size", type=int, default=200)
    p_back.add_argument("--output", default="outputs/backtest.csv")
    p_back.set_defaults(func=cmd_backtest)

    p_train = sub.add_parser("train", help="训练并输出概率摘要")
    p_train.add_argument("--model", choices=["frequency", "rolling", "decay", "bayesian", "ensemble", "pymc"], default="bayesian")
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="生成候选组合")
    p_predict.add_argument("--model", choices=["frequency", "rolling", "decay", "bayesian", "ensemble", "pymc"], default="ensemble")
    p_predict.add_argument("--top-k", type=int, default=20)
    p_predict.add_argument("--seed", type=int, default=20260529)
    p_predict.add_argument("--pool-size", type=int, default=2000)
    p_predict.set_defaults(func=cmd_predict)

    p_report = sub.add_parser("report", help="生成 Markdown 报告与 CSV 附件")
    p_report.add_argument("--model", choices=["frequency", "rolling", "decay", "bayesian", "ensemble"], default="bayesian")
    p_report.add_argument("--min-train-size", type=int, default=200)
    p_report.add_argument("--output-dir", default="outputs")
    p_report.set_defaults(func=cmd_report)

    p_pymc_fit = sub.add_parser("pymc-fit", help="拟合 PyMC 贝叶斯后验模型")
    p_pymc_fit.add_argument("--alpha", type=float, default=1.0)
    p_pymc_fit.add_argument("--beta", type=float, default=1.0)
    p_pymc_fit.add_argument("--draws", type=int, default=500)
    p_pymc_fit.add_argument("--tune", type=int, default=500)
    p_pymc_fit.add_argument("--chains", type=int, default=2)
    p_pymc_fit.add_argument("--cores", type=int, default=2)
    p_pymc_fit.add_argument("--seed", type=int, default=20260529)
    p_pymc_fit.add_argument("--target-accept", type=float, default=0.9)
    p_pymc_fit.add_argument("--sampler", choices=["pymc", "numpyro"], default="pymc")
    p_pymc_fit.add_argument("--quick", action="store_true", help="只使用最近 500 期做快速验证")
    p_pymc_fit.add_argument("--no-sample", action="store_true", help="跳过 MCMC，仅输出共轭后验解析均值")
    p_pymc_fit.set_defaults(func=cmd_pymc_fit)

    p_pymc_predict = sub.add_parser("pymc-predict", help="使用 PyMC 贝叶斯模型生成候选组合")
    p_pymc_predict.add_argument("--alpha", type=float, default=1.0)
    p_pymc_predict.add_argument("--beta", type=float, default=1.0)
    p_pymc_predict.add_argument("--draws", type=int, default=500)
    p_pymc_predict.add_argument("--tune", type=int, default=500)
    p_pymc_predict.add_argument("--chains", type=int, default=2)
    p_pymc_predict.add_argument("--cores", type=int, default=2)
    p_pymc_predict.add_argument("--seed", type=int, default=20260529)
    p_pymc_predict.add_argument("--target-accept", type=float, default=0.9)
    p_pymc_predict.add_argument("--sampler", choices=["pymc", "numpyro"], default="pymc")
    p_pymc_predict.add_argument("--quick", action="store_true", help="只使用最近 500 期做快速验证")
    p_pymc_predict.add_argument("--no-sample", action="store_true", help="跳过 MCMC，仅使用解析后验均值")
    p_pymc_predict.add_argument("--top-k", type=int, default=20)
    p_pymc_predict.add_argument("--pool-size", type=int, default=2000)
    p_pymc_predict.set_defaults(func=cmd_pymc_predict)

    p_mac = sub.add_parser("mac-bootstrap", help="打印 Apple Silicon macOS 部署命令")
    p_mac.add_argument("--print-commands", action="store_true", help="兼容脚本/测试；默认即打印命令")
    p_mac.set_defaults(func=cmd_mac_bootstrap)

    p_web = sub.add_parser("web", help="启动本地 Streamlit Web App")
    p_web.add_argument("--server-port", type=int, default=8501)
    p_web.add_argument("--print-command", action="store_true", help="只打印启动命令，不实际启动")
    p_web.set_defaults(func=cmd_web)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

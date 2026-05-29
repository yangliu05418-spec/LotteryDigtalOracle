# 双色球概率预测系统（学术概率建模）

> **学术用途声明**：本项目仅用于组合数学、概率分布、统计检验、时间序列回测和模型解释研究。输出内容不构成投注建议，不承诺命中，不保证现实预测能力，也不讨论投注金额或收益方案。

## 项目定位

本系统现在面向 **Apple Silicon macOS / M4 Mac mini** 做完整部署优化，并将 **PyMC 贝叶斯建模** 与 **Streamlit 本地 Web App** 作为完整研究环境的推荐能力。它把双色球历史开奖数据视为离散组合空间样本，先建立理论随机基准与统计检验，再提供频率模型、滚动窗口模型、指数衰减模型、轻量贝叶斯平滑模型、PyMC 后验模型、集成模型、rolling-origin 回测、候选组合生成和浏览器交互界面。

## 全新 M4 Mac mini 一键部署与启动 Web App

在一台几乎全新的 Apple Silicon Mac 上，进入项目目录后运行：

```bash
chmod +x scripts/install_and_run_macos.sh
./scripts/install_and_run_macos.sh
```

这个脚本会自动完成：

1. 检查 macOS 与 `arm64` 架构。
2. 若没有 conda，则自动下载安装 Apple Silicon Miniforge。
3. 创建或更新 `ssq-model-macos-arm64` 环境。
4. 安装 PyMC、JAX、NumPyro、BlackJAX、Streamlit、Plotly 和项目包。
5. 运行测试、基础分析和 PyMC 快速验证。
6. 启动本地 Streamlit Web App：

```text
http://127.0.0.1:8501
```

以后日常启动只需：

```bash
./scripts/run_web_macos.sh
```

也可以在已激活环境中运行：

```bash
python3 -m ssq_model web
```

Web App 是本机浏览器界面，默认只监听 `127.0.0.1`，用于本地概率建模研究。

核心边界：

- 研究对象是概率分布、历史样本偏差、模型评分与随机基准对照。
- 候选组合只是模型输出的概率性样本集合。
- 历史统计偏差不能直接推出未来开奖规律。
- 禁止将结果表述为“稳赚”“必中”“必出”等保证性预测。

## 环境安装

在 macOS 中请优先使用 `python3`。全新 M4 Mac mini 推荐直接运行 Web 一键部署脚本：

```bash
chmod +x scripts/install_and_run_macos.sh
./scripts/install_and_run_macos.sh
```

该脚本会安装/复用 Miniforge conda 环境，安装 `.[macos-arm64,web]`，验证 PyMC / ArviZ / JAX / NumPyro / BlackJAX / Streamlit / Plotly 导入，并启动本地 Web App。

当前核心代码仍保持标准库可运行，便于先做基础检查：

```powershell
python3 -m unittest discover -s tests -v
python3 -m ssq_model analyze
```

如需后续图表、机器学习、SciPy 精确检验等扩展，可安装基础研究依赖：

```powershell
python -m pip install -r requirements.txt
```

完整 Apple Silicon 贝叶斯环境：

```powershell
python3 -m pip install -e '.[macos-arm64,web]'
```

## GPU 非必需 / PyMC 是否必须

- **GPU 非必需**：当前约 3457 期历史样本，M4 Mac mini 的 CPU 足以运行主要统计、回测和 PyMC 小样本 NUTS 采样。
- **PyMC 是完整研究环境的推荐必装模块**：用于蓝球 Dirichlet 后验、红球 Beta-Binomial 边际后验、MCMC 采样和后验不确定性研究。
- JAX/NumPyro/BlackJAX 作为 Apple Silicon 上的可选采样后端，若遇到依赖问题可回退默认 PyMC sampler。

## 数据更新方式

默认数据文件：

```text
数据/历史数据.csv
```

可使用已有爬虫更新：

```powershell
python3 ./crawler_ssq.py
```

CSV 标准字段：

```text
期号, 日期, 红球-1, 红球-2, 红球-3, 红球-4, 红球-5, 红球-6, 蓝球
```

建模前会校验：期号唯一、日期可解析、红球 1..33 且 6 个无重复、蓝球 1..16。

## 常用命令

基础统计画像与理论组合空间：

```powershell
python3 -m ssq_model analyze
```

随机性 / 均匀性检验：

```powershell
python3 -m ssq_model test-randomness
```

rolling-origin 回测（严格只使用目标期之前的数据）：

```powershell
python3 -m ssq_model backtest --model bayesian --min-train-size 200
```

训练并查看概率摘要：

```powershell
python3 -m ssq_model train --model ensemble
```

生成候选组合（仅为概率模型输出）：

```powershell
python3 -m ssq_model predict --top-k 20
```

生成 Markdown 报告和 CSV 附件：

```powershell
python3 -m ssq_model report --output-dir outputs
```

PyMC 贝叶斯快速验证：

```bash
python3 -m ssq_model pymc-fit --quick --no-sample
```

PyMC MCMC 采样：

```bash
python3 -m ssq_model pymc-fit --draws 500 --tune 500 --chains 2 --cores 2
```

PyMC 候选组合：

```bash
python3 -m ssq_model pymc-predict --top-k 20 --no-sample
```

启动本地 Web App：

```bash
python3 -m ssq_model web
```

只打印 Streamlit 启动命令：

```bash
python3 -m ssq_model web --print-command
```

## 模型说明

已实现模型：

- `frequency`：历史边际频率模型。
- `rolling`：滚动窗口频率模型。
- `decay`：指数衰减频率模型。
- `bayesian`：轻量贝叶斯平滑模型（默认推荐作为稳定基线）。
- `pymc`：PyMC 后验模型（红球 Beta-Binomial 边际 + 蓝球 Dirichlet）。
- `ensemble`：多模型线性集成。

所有模型输出：

- 红球 1..33 的边际出现概率，概率和约为 6。
- 蓝球 1..16 的分类概率，概率和为 1。

## 回测指标

首要关注概率分布质量：

- 红球边际 log loss
- 蓝球 log loss
- 红球 Brier score
- 蓝球 Brier score

辅助观察候选组合命中数、Top-k 表现等，但必须与理论随机基准共同解读。

## 输出文件说明

`python -m ssq_model report` 默认生成：

```text
outputs/report.md      # Markdown 研究报告
outputs/features.csv   # 派生特征表
outputs/backtest.csv   # 逐期 rolling-origin 回测结果
```

`python -m ssq_model backtest --output outputs/backtest.csv` 会写出逐期回测 CSV。

## 结果如何解释

推荐解释顺序：

1. 先说明数学定义或统计假设。
2. 再说明计算方法和样本范围。
3. 给出与随机基准的比较。
4. 最后说明限制：历史偏差不代表可利用的未来规律。

示例表述：

- “在当前历史样本中观察到……”
- “该结果相对于随机基准……”
- “该偏差是否显著需要通过统计检验确认。”
- “这不能直接推出未来开奖规律。”

禁止性表述：

- “下一期必然……”
- “一定会出……”
- “稳赚……”
- “必中……”

## 开发与测试

```powershell
python3 -m unittest discover -s tests -v
```

当前测试覆盖数据校验、理论组合空间、特征计算、模型概率合法性、候选组合生成、回测无未来信息泄漏和文档边界声明。

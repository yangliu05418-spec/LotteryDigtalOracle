# Apple Silicon macOS / M4 Mac mini 部署指南

本项目现在以 **Apple Silicon macOS + PyMC 贝叶斯建模** 为主要部署目标。完整贝叶斯工作流推荐在 M4 Mac mini 上运行；命令统一使用 macOS 常见的 `python3`。

> 学术用途声明：本项目只用于概率分布、统计检验、贝叶斯后验和历史回测研究，不构成投注建议，不承诺命中。

## 方案 0：Streamlit Web App 一键部署（推荐给全新 Mac）

在一台全新的 M4 Mac mini 上，进入项目目录后运行：

```bash
chmod +x scripts/install_and_run_macos.sh
./scripts/install_and_run_macos.sh
```

脚本会：

- 检查 `Darwin` / `arm64`
- 若没有 conda，自动下载并安装 `Miniforge3-MacOSX-arm64.sh`
- 创建或更新 `ssq-model-macos-arm64`
- 安装 `.[macos-arm64,web]`
- 验证 PyMC / JAX / Streamlit / Plotly
- 启动：

```bash
streamlit run ssq_model/web_app.py --server.address=127.0.0.1 --server.port=8501
```

以后日常启动：

```bash
./scripts/run_web_macos.sh
```

或：

```bash
python3 -m ssq_model web
```

## 方案 1：venv 一键部署（轻量 CLI 备用）

在项目根目录运行：

```bash
chmod +x scripts/bootstrap_macos.sh
./scripts/bootstrap_macos.sh
```

脚本会执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e '.[macos-arm64]'
python3 -c "import pymc, arviz, jax, numpyro, blackjax; print('PyMC Apple Silicon environment OK')"
python3 -m unittest discover -s tests -v
python3 -m ssq_model analyze
python3 -m ssq_model pymc-fit --quick --no-sample
```

脚本会检查：

- `uname -s` 必须是 `Darwin`
- `uname -m` 必须是 `arm64`
- `python3` 必须存在

## 方案 2：Miniforge / conda-forge 环境

如果你偏好 conda-forge 科学计算栈：

```bash
conda env create -f environment-macos-arm64.yml
conda activate ssq-model-macos-arm64
python3 -m ssq_model analyze
python3 -m ssq_model pymc-fit --quick --no-sample
```

## PyMC 命令

快速解析后验均值，不跑 MCMC：

```bash
python3 -m ssq_model pymc-fit --quick --no-sample
```

运行 PyMC 默认 NUTS 采样：

```bash
python3 -m ssq_model pymc-fit --draws 500 --tune 500 --chains 2 --cores 2
```

使用 NumPyro/JAX 采样后端：

```bash
python3 -m ssq_model pymc-fit --sampler numpyro --draws 500 --tune 500 --chains 2
```

使用贝叶斯后验均值生成候选组合：

```bash
python3 -m ssq_model pymc-predict --top-k 20 --no-sample
```

## M4 Mac mini 参数建议

- 快速验证：`--quick --no-sample`
- 日常研究：`--draws 500 --tune 500 --chains 2 --cores 2`
- 更稳定后验：`--draws 1000 --tune 1000 --chains 4 --cores 4`
- JAX/NumPyro 后端：可试 `--sampler numpyro`，若遇到依赖问题，回退默认 `--sampler pymc`。

## 故障排查

如果 `python3` 不存在，请先安装 Apple Silicon 版本 Python 或 Miniforge。

如果 PyMC 导入失败，重新执行：

```bash
source .venv/bin/activate
python3 -m pip install -e '.[macos-arm64]'
python3 -c "import pymc, arviz, jax, numpyro, blackjax; print('ok')"
```

如果 JAX/NumPyro 采样失败，先使用默认 PyMC 采样：

```bash
python3 -m ssq_model pymc-fit --sampler pymc --draws 500 --tune 500 --chains 2 --cores 2
```

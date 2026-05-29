# PyMC Apple Silicon Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PyMC Bayesian models and macOS Apple Silicon deployment packaging for the SSQ probability research system.

**Architecture:** Keep the existing lightweight models intact, but add a focused `pymc_models.py` optional backend that imports PyMC lazily and fails with clear install guidance if unavailable. CLI exposes PyMC commands separately, and macOS deployment uses `python3`, `venv`, and Apple Silicon checks.

**Tech Stack:** Python 3.11+, PyMC, ArviZ, optional JAX/NumPyro/BlackJAX, unittest, setuptools, macOS shell scripts.

---

### Task 1: PyMC backend tests

**Files:**
- Create: `tests/test_pymc_backend.py`
- Modify: none

- [ ] Write tests for lazy import errors, analytic posterior predictions without sampling, and model factory names.
- [ ] Run `python -m unittest tests.test_pymc_backend -v`; expect failure because `ssq_model.models.pymc_models` does not exist.
- [ ] Implement minimal backend.
- [ ] Run test again; expect pass.

### Task 2: PyMC backend implementation

**Files:**
- Create: `ssq_model/models/pymc_models.py`
- Modify: `ssq_model/models/__init__.py`

- [ ] Add `PyMCNotInstalledError` with macOS install command text.
- [ ] Add `PyMCBlueDirichletModel` with analytic posterior mean and optional MCMC sampling.
- [ ] Add `PyMCRedBetaBinomialModel` with analytic posterior mean and optional MCMC sampling.
- [ ] Add `PyMCJointBayesianModel` that combines red and blue posterior predictions.
- [ ] Export classes from `models/__init__.py`.

### Task 3: CLI tests and implementation

**Files:**
- Modify: `tests/test_ssq_model_core.py` or create `tests/test_cli_pymc.py`
- Modify: `ssq_model/cli.py`

- [ ] Test parser includes `pymc-fit`, `pymc-predict`, `mac-bootstrap`.
- [ ] Add CLI commands that use `python3`-oriented mac install guidance.
- [ ] Keep default `train/backtest/predict` compatible with existing models.

### Task 4: macOS deployment assets

**Files:**
- Create: `environment-macos-arm64.yml`
- Create: `scripts/bootstrap_macos.sh`
- Create: `docs/MACOS_APPLE_SILICON.md`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] Environment YAML includes Python 3.11, pymc, arviz, numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, statsmodels, jax, numpyro, blackjax.
- [ ] Bootstrap script uses Bash, `python3 -m venv`, `python3 -m pip`, architecture checks for arm64, and smoke tests imports.
- [ ] Docs state project is Mac-first, PyMC recommended/required for full Bayesian workflow, commands use `python3`.
- [ ] Package extras include `macos-arm64` and keep `bayesian-extra` alias.

### Task 5: Verification

**Files:** all touched files

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m compileall -q ssq_model tests`.
- [ ] Run `python -m ssq_model analyze`.
- [ ] Run `python -m ssq_model pymc-fit --quick --no-sample`.
- [ ] Run `python -m ssq_model mac-bootstrap --print-commands`.

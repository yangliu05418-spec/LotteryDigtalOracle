# Streamlit Local Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit local web app and one-click Apple Silicon macOS deployment flow.

**Architecture:** Keep UI code thin in `web_app.py`, put testable logic in `ssq_model/web/` helper modules, and launch Streamlit through CLI/subprocess only when requested. macOS scripts install or reuse Miniforge/conda and start the app with `streamlit run`.

**Tech Stack:** Python 3.11+, Streamlit, Plotly, PyMC/JAX stack, Bash, unittest.

---

### Task 1: Tests

**Files:**
- Create: `tests/test_web_app.py`
- Create: `tests/test_web_deploy.py`
- Modify: `tests/test_documentation.py`

- [ ] Write tests for web helper imports, dashboard summary, chart fallback specs, CLI `web`, mac scripts, pyproject extras, and README one-click instructions.
- [ ] Run `python -m unittest tests.test_web_app tests.test_web_deploy -v` and confirm it fails because modules/scripts do not exist.

### Task 2: Web modules

**Files:**
- Create: `ssq_model/web/__init__.py`
- Create: `ssq_model/web/state.py`
- Create: `ssq_model/web/charts.py`
- Create: `ssq_model/web/actions.py`
- Create: `ssq_model/web_app.py`

- [ ] Implement pure helpers first.
- [ ] Implement Streamlit UI with lazy imports.
- [ ] Keep all public-facing text academically bounded.

### Task 3: CLI launch

**Files:**
- Modify: `ssq_model/cli.py`

- [ ] Add `streamlit_launch_command()` helper returning `python3 -m streamlit run ssq_model/web_app.py`.
- [ ] Add `web` subcommand with `--print-command` and `--server-port`.
- [ ] Launch via subprocess only when not printing.

### Task 4: macOS deployment

**Files:**
- Create: `scripts/install_and_run_macos.sh`
- Create: `scripts/run_web_macos.sh`
- Modify: `environment-macos-arm64.yml`
- Modify: `pyproject.toml`

- [ ] Add `web` extra with Streamlit and Plotly.
- [ ] Include Streamlit and Plotly in `macos-arm64` and conda YAML.
- [ ] Script checks macOS arm64, installs Miniforge if conda missing, creates env, installs project, tests, launches app.
- [ ] Daily launch script activates env and runs Streamlit.

### Task 5: README/docs

**Files:**
- Modify: `README.md`
- Modify: `docs/MACOS_APPLE_SILICON.md`

- [ ] Put one-click deployment commands near the top of README.
- [ ] Document daily launch and `python3 -m ssq_model web`.
- [ ] Explain that the Web app is local-only and academic-use only.

### Task 6: Verification

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m compileall -q ssq_model tests`.
- [ ] Run `python -m ssq_model web --print-command`.
- [ ] Run `python -m ssq_model analyze`.

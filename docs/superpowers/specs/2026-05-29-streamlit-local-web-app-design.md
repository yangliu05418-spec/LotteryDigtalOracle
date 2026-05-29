# Streamlit Local Web App Design

## Goal

Add a local Streamlit web interface for the SSQ probability research system and provide a one-command Apple Silicon macOS installer/launcher for a fresh M4 Mac mini.

## Scope

- Add a Streamlit app with dashboard, data/statistics, PyMC Bayesian modeling, candidate generation, and report center tabs.
- Add `python3 -m ssq_model web` to launch `streamlit run ssq_model/web_app.py`.
- Add macOS one-time installer `scripts/install_and_run_macos.sh` and daily launcher `scripts/run_web_macos.sh`.
- Update `pyproject.toml`, `environment-macos-arm64.yml`, `docs/MACOS_APPLE_SILICON.md`, and `README.md`.
- Keep academic-purpose disclaimers visible in Web and CLI flows.

## Architecture

- `ssq_model/web/state.py`: environment and data summary helpers, no Streamlit dependency.
- `ssq_model/web/charts.py`: chart builders with lazy Plotly import and testable fallback chart specs.
- `ssq_model/web/actions.py`: reusable actions for analysis, PyMC prediction, candidates, and reports.
- `ssq_model/web_app.py`: Streamlit UI layer only; imports Streamlit lazily when run.
- Existing CLI remains the stable automation interface.

## Deployment

The fresh Mac path is:

```bash
chmod +x scripts/install_and_run_macos.sh
./scripts/install_and_run_macos.sh
```

The script checks Darwin/arm64, installs Miniforge if needed, creates/updates the conda env from `environment-macos-arm64.yml`, installs `.[macos-arm64,web]`, runs smoke tests, then launches Streamlit.

Daily launch:

```bash
./scripts/run_web_macos.sh
```

## Testing

Use unittest without requiring Streamlit/Plotly installed locally. Tests verify pure helper behavior, CLI parser support, script contents, pyproject extras, and README deployment instructions.

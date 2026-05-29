#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ssq-model-macos-arm64"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher is intended for macOS only." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  else
    echo "Conda/Miniforge not found. Run scripts/install_and_run_macos.sh first." >&2
    exit 1
  fi
else
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda activate ssq-model-macos-arm64
streamlit run ssq_model/web_app.py --server.address=127.0.0.1 --server.port=8501

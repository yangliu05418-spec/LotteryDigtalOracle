#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ssq-model-macos-arm64"
MINIFORGE_DIR="$HOME/miniforge3"
MINIFORGE_INSTALLER="Miniforge3-MacOSX-arm64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${MINIFORGE_INSTALLER}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is intended for macOS only." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This installer is optimized for Apple Silicon arm64 Macs. Detected: $(uname -m)" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Installing Miniforge for Apple Silicon..."
  TMP_DIR="$(mktemp -d)"
  curl -L "$MINIFORGE_URL" -o "$TMP_DIR/$MINIFORGE_INSTALLER"
  bash "$TMP_DIR/$MINIFORGE_INSTALLER" -b -p "$MINIFORGE_DIR"
  # shellcheck disable=SC1091
  source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
else
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing conda environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f environment-macos-arm64.yml --prune
else
  echo "Creating conda environment: $ENV_NAME"
  conda env create -f environment-macos-arm64.yml
fi

conda activate "$ENV_NAME"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e '.[macos-arm64,web]'
python3 -c "import streamlit, plotly, pymc, arviz, jax, numpyro, blackjax; print('Streamlit + PyMC Apple Silicon environment OK')"
python3 -m unittest discover -s tests -v
python3 -m ssq_model analyze
python3 -m ssq_model pymc-fit --quick --no-sample

echo "Launching local Web App at http://127.0.0.1:8501"
streamlit run ssq_model/web_app.py --server.address=127.0.0.1 --server.port=8501

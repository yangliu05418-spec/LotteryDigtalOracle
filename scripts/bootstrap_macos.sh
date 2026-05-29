#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This bootstrap script is intended for macOS only." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This bootstrap script is optimized for Apple Silicon arm64 Macs." >&2
  echo "Detected architecture: $(uname -m)" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Install Python 3.11+ or Miniforge for Apple Silicon first." >&2
  exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e '.[macos-arm64]'
python3 -c "import pymc, arviz, jax, numpyro, blackjax; print('PyMC Apple Silicon environment OK')"
python3 -m unittest discover -s tests -v
python3 -m ssq_model analyze
python3 -m ssq_model pymc-fit --quick --no-sample

echo "SSQ PyMC environment is ready. Activate it later with: source .venv/bin/activate"

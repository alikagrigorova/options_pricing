#!/bin/bash
# One-time setup on the SCC: Python virtual environment with the dependencies.
#   bash scc/setup_env.sh            (run from the repository root, on a login node)
set -euo pipefail
module load python3/3.10.12 2>/dev/null || module load python3
VENV=${VENV:-$HOME/venvs/simgreeks}
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt pyarrow
python -m pytest -q tests
echo "environment ready: $VENV"

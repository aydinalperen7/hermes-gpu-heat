#!/usr/bin/env bash
set -euo pipefail

# Run from repo root by default.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Optional overrides.
ENV_NAME="${ENV_NAME:-hermes_clean}"
CONFIG_PATH="${1:-sim_ex1.ini}"
LASER_PATH="${2:-path_laser_ex1.ini}"
PRECOND_LEVEL2="${PRECOND_LEVEL2:-${3:-none}}"
PRECOND_LEVEL3="${PRECOND_LEVEL3:-${4:-none}}"
EXTRA_ARGS=("${@:5}")

case "$PRECOND_LEVEL2" in
  none|jacobi) ;;
  *)
    echo "Invalid preconditioner for level2: $PRECOND_LEVEL2 (expected: none|jacobi)" >&2
    exit 2
    ;;
esac

case "$PRECOND_LEVEL3" in
  none|jacobi) ;;
  *)
    echo "Invalid preconditioner for level3: $PRECOND_LEVEL3 (expected: none|jacobi)" >&2
    exit 2
    ;;
esac

# Initialize conda without relying on interactive shell dotfiles.
if ! command -v conda >/dev/null 2>&1; then
  source ~/.bashrc >/dev/null 2>&1 || true
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)"
else
  echo "conda command not found in PATH" >&2
  exit 1
fi

conda activate "$ENV_NAME"
module purge
module load cuda
module load gcc

export CUDA_HOME="${TACC_CUDA_DIR:-/home1/apps/nvidia/Linux_aarch64/24.7/cuda/12.5}"
export NUMBA_CUDA_DRIVER=/usr/lib64/libcuda.so
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/12.5/targets/sbsa-linux/lib:/usr/lib64:${LD_LIBRARY_PATH:-}"

python -u -X faulthandler src/hermes/scripts/multi_level_solver.py \
  --config "$CONFIG_PATH" \
  --laser_path "$LASER_PATH" \
  --precond-level2 "$PRECOND_LEVEL2" \
  --precond-level3 "$PRECOND_LEVEL3" \
  "${EXTRA_ARGS[@]}"

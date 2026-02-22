#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-hermes_clean}"

if ! command -v conda >/dev/null 2>&1; then
  source ~/.bashrc >/dev/null 2>&1 || true
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)"
else
  echo "conda command not found in PATH" >&2
  exit 1
fi

export CONDA_NO_PLUGINS=true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f environment.gh-node.yml --prune
else
  conda env create -f environment.gh-node.yml
fi

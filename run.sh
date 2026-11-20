#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VTOOLS_PYTHON="$HOME/python-envs/vtools/.venv/bin/python"
exec "$ROOT/run-python.sh" "$ROOT/launcher.py" "$@"

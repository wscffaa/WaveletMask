#!/usr/bin/env bash
# Reuse the shared rtn conda env on servers; Mac uses local python.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH=$PYTHONPATH"
python3 - <<'PY'
import basicofr, basicsr
print("BasicOFR env OK:", basicofr.__file__)
PY

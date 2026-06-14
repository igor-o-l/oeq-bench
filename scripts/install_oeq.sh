#!/usr/bin/env bash
set -euo pipefail

PACKAGE_SPEC="${1:-openequivariance>=0.6,<0.7}"

export CXXFLAGS="${CXXFLAGS:-} -include cstdint"

python -m pip install "${PACKAGE_SPEC}"
python - <<'PY'
import openequivariance as oeq
print("OpenEquivariance import OK:", getattr(oeq, "__version__", "unknown"))
PY

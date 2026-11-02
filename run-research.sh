#!/bin/sh
# Keep this checkout independent of another conversation's temporary venv.
FUKAJI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
export PYTHONPATH="$FUKAJI_ROOT/build/runtime-deps:$FUKAJI_ROOT:$FUKAJI_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
FUKAJI_PYTHON=${FUKAJI_PYTHON:-python3}
"$FUKAJI_PYTHON" -c 'import importlib.metadata as m; import numpy; assert numpy.__version__=="2.3.3" and m.version("python-shogi")=="1.1.1", "Install requirements.txt into build/runtime-deps before research"' || exit 2
cd "$FUKAJI_ROOT" || exit 2
exec "$FUKAJI_PYTHON" "$@"

#!/bin/sh
# Resolve code and pinned model independently of the GUI working directory.
FUKAJI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
export PYTHONPATH="$FUKAJI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${FUKAJI_PYTHON:-python3}" -m fukaji.engine "$@"

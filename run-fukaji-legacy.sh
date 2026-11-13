#!/bin/sh
# 旧正式エンジン(深路の自前探索 v0.4.1)。比較・研究ワークフロー用。
FUKAJI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
export PYTHONPATH="$FUKAJI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${FUKAJI_PYTHON:-python3}" -m fukaji.engine "$@"

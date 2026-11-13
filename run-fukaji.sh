#!/bin/sh
# 正式エンジン(2026-11-13〜): ハイブリッド＝雲路の探索＋深路の正式NNUE評価。
# 旧エンジン(深路の自前探索 v0.4.1)は FUKAJI_ENGINE=legacy で起動できる。
# 初回は雲路の取得とビルドが走る。GUIで使う前に: python3 scripts/run_hybrid.py --prepare
FUKAJI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
export PYTHONPATH="$FUKAJI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ "$FUKAJI_ENGINE" = "legacy" ]; then
  exec "${FUKAJI_PYTHON:-python3}" -m fukaji.engine "$@"
fi
exec "${FUKAJI_PYTHON:-python3}" "$FUKAJI_ROOT/scripts/run_hybrid.py" "$@"

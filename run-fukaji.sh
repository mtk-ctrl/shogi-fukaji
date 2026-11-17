#!/bin/sh
# 正式エンジン(2026-11-13〜): ハイブリッド＝雲路の探索＋深路の正式NNUE評価。
# 初回は雲路の取得とビルドが走る。GUIで使う前に: python3 scripts/run_hybrid.py --prepare
# (旧エンジン=深路の自前探索 v0.4.1 の起動口は2026-11-17に削除)
FUKAJI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
export PYTHONPATH="$FUKAJI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${FUKAJI_PYTHON:-python3}" "$FUKAJI_ROOT/scripts/run_hybrid.py" "$@"

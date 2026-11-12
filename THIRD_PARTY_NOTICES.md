# サードパーティ通知 / Third-party notices

## ライセンス
深路(FUKAJI)のコードは **GNU General Public License v3.0**（`LICENSE`）で公開します。
NNUE重み・研究データ(`models/`、`docs/` 等)は、特に断りがない限り同じ条件で扱います。

## 依存・参照している GPL コード
- **python-shogi 1.1.1**（https://github.com/gunyarakun/python-shogi 、GPL-3.0）: 将棋のルール・局面処理に `import shogi` で使用。
- **雲路(KUMOJI)**（https://github.com/mtk-ctrl/shogi-ai 、GPL-3.0）: 対局相手・USI通信部品・ハイブリッド実験（`research/hybrid/`, `scripts/build_hybrid.py`）。雲路のソースはこのリポジトリには含めず、CI でビルド時に取得して差し込む。雲路自体が、やねうら王(YaneuraOu、GPL-3.0、https://github.com/yaneurao/YaneuraOu)由来のルール実装を含む。詳細は雲路の `THIRD_PARTY_NOTICES.md`。
- **numpy 2.3.3**（BSD-3-Clause）。

ハイブリッドエンジン等、雲路のコードと一体でビルドしたバイナリを配布する場合は、GPL-3.0 に従い対応するソース（雲路のソース・本リポジトリ・パッチ・ビルド手順）を併せて提供すること。

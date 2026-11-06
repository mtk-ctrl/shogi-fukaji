# D44：D43 NNUEを現行採用 — GitHub反映・配布・実起動まで検証完了

更新日：2026-11-07（JST）。

## 採用前後

| 項目 | 旧 | 採用済みの現行 |
|---|---|---|
| NNUE研究snapshot SHA256 | `170d6c1ee41d77b69f0b75722ece3deb85f3eea189f70d9789db477a38266db1`（D37） | `bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f`（D43） |
| Adam step | 2106 | **2882** |
| 追加学習の材料 | — | 新規自己対局1000局（994決着、6未決着）＋既存train665局 |
| 標準棋力比較 | — | D43 vs D37、**200局：150勝50敗・引分0・未決着0** |
| 推論用weights | 過去配布を保護 | **`models/fukaji-d43.nnue.npz`**、SHA256 `f6ba1d5af46ae648b1921d675ada28cf5065f23a20e424154458817534bf42a6` |
| 正式USIの版名/探索仕様 | `FUKAJI v0.4.0` / 既定100ms / C++ NNUE q0 | **変更しない（F00の区分に従う）** |

## 実行・検証証拠

1. [D43 実研究run 42292240321](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42292240321)：準備＋20個の自己対局ランナー×5wave＋集約＋単一学習＋比較20ランナー＋最終集計、**全125 job success**。学習1000局、D43重み `bf8c26f9…`、Adam2106→2882、200局比較150対50。学習後重みと完全optimizerは[学習Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d43-d37-trained-42292240321)（SHA256 `955a1785940f9f2aac946739af1b8637e5ef53d06297e6ad7d32b75a5a94a236`）、200局の原棋譜・事前planは[比較Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d43-d37-before-after-200games-42292240321)（SHA256 `c08ccab3f5e9ef90c288e8ccfc5f3951eb188dfc30876af775c2ba3cfb03f510`）。
2. [D44 現行化run 42309719933](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42309719933)：**completed / success**。D43学習Releaseと比較ReleaseのSHA-256が両方一致し、全200棋譜をアーカイブ上の個別SHAへ再照合。学習済みNNUE＋Adam2882を読み、NNUEの4推論テンソルを正確に抽出・照合してoptimizerを含まないファイルを作成。
3. 正式仕様と契約を整合させ、**713件の回帰テスト成功**（D44 runログの `Ran 713 tests in 112.675s`）。`scripts/check_fukaji_contracts.py`は `Formal contract, router links, durable state and explicit resume/publication: OK`。実C++ NNUEの探索一致・USI応答、指し手合法性、起動中断の検証に成功。
4. **mainの現行モデル更新commit `28f9d7ef0f3b77ac9086adae50e5b1f1162667e4`**：[GitHubコミット](https://github.com/mtk-ctrl/shogi-fukaji/commit/28f9d7ef0f3b77ac9086adae50e5b1f1162667e4)。`formal/model.json`、Git管理下の推論用binary、`research/d17-fast-self-learning/formal-contract.json`、`docs/現行エンジン仕様.md`、`AGENTS.md`、開発ルーター、ROADMAP、[採用判断](decisions/2026-11-07_D43学習後モデルを現行に採用.md)を**同一commit**で反映。GitHubから `formal/model.json` を再取得し D43 SHA／Adam2882／推論重みSHAを確認。
5. [D44 現行モデル配布Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-current-d43-20261107)：推論用重み＋エンジンソースを同梱した `D44-FUKAJI-current-D43.tar.gz`（620,666 bytes、SHA256 **`3957b46cef858d9d0637723d927e847d9c00c11f098973d27ac69e65a26b81b0`**）を公開。**別領域へ展開し、BUNDLE.jsonで構成ファイルハッシュ検証、実USI／C++ NNUEの起動試験成功**。Releaseから再ダウンロードした3資産（bundle・動作receipt・採用receipt）について実ファイルとSHA256一致を確認。GitHub Release画面の資産digestと一致。

## 維持する境界・再開点

- D44で変えたのは**現行NNUEの重み・参照先・Adam学習起点・採用記録**。バイナリの推論テンソル形式、正式USI identity `FUKAJI v0.4.0`、探索コード、思考時間の既定値は変更していない。版名変更は本人の別指示事項であり、この結果は自動的な版名更新ではない。
- **次周の唯一の学習親はD43 snapshot `bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f` / Adam2882**。更新前snapshotとしてこのD43を凍結し、更新後を直接比較する。次周に失敗すればD43へ復元する（D37へ戻さない）。
- D37 / Adam2106は歴史的・復元用に保管。既存D41等の不採用学習後重みは次回の親にしない。将来の採用時にはこのD44手順に従い、**契約更新と実起動検証が完了する前にmainを更新しない**。
- 機密境界は本人の2026-11-05公開許可を適用。GitHubを重み・optimizer・棋譜の現行正本として運用し、Driveの歴史的資産は移行検証が終わるまで保持する。

## 既存資料の読み方

`docs/ROADMAP.md`と過去D系文書には「D37が現行」「D43未実行」「D43採用保留」という**その日付時点の記録**が保存されている。2026-11-07以後の現在値は本D44実証と`formal/model.json`が正本。D37を次周の親として復元しない。

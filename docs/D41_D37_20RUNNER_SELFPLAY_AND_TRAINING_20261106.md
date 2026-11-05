# D41：D37の20ランナー自己対局40局とAdam継続学習

更新日：2026-11-06（JST）。**自己対局と学習・GitHub Release保全が完了。棋力比較は未実施。現行モデルはD37のまま。**

## 目的と条件

本人の指示：「20ランナー使って自己対局で強化してください。ルーターとルールを読んで従ってください。」

- 現行モデルは **D37（FUKAJI v0.4.0）**、学習前SHA256 `170d6c1ee41d77b69f0b75722ece3deb85f3eea189f70d9789db477a38266db1`、Adam **2106**。旧D39は親にしない。
- 通常1手200ms、1秒長考は各陣営最大10回、C++ NNUE・静止探索0、各局上限512手。
- 20独立GitHub Actions jobs × 2局。40個の独立序盤（自己対局では序盤ペアの先後入替ではなく、親ID重複を避けて分離）。序盤直後24手に探索的着手5%の設定。
- 公開GitHubをソース・学習状態・棋譜・検証結果の正本とする2026-11-05本人決定に従う。旧Drive成果はGitHub移行済みと検証できるまでは損失防止のため保持。

## 成果1：20ランナー自己対局

GitHub Actions [run **42017791719**](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42017791719)、2026-11-05 11:17〜11:20 UTC。

- **prepare 1、selfplay 20、aggregate 1：計22 jobs全件success**。20 selfplayの独立job IDをGitHub RESTから確認。計画上の並列値だけではない。
- **40局/40局完了、40局詰み決着、全5,351手**、長考使用781手。各laneのゲームを集約ジョブが別途合法検証。
- 完全原棋譜・検証レポートを[公開Release `fukaji-d41-d37-selfplay-42017791719`](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d41-d37-selfplay-42017791719)に保全。
- Release archive `D41-D37-20runner-selfplay-40games.tar.gz`、**SHA256 `8708db460d4d2bcad949d91ab881d9aeb804abcb8e28808dbbb70bda5c21bd9f`**。学習側でReleaseからダウンロードしてこのSHAの一致を再確認。
- 学習前の公開学習状態 `research/d41-public/D37-Adam2106-training-state.tar.gz`、SHA256 `41c8f56b76cd5d7aa87bff9131d553b304fa4134bf7448c05d3309458f66760e`。学習前状態が元D37と合致することをverify。

## 成果2：40局からのD37直接継続学習

最初のGitHub Actions [run **42018509500**](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42018509500)は失敗。D37と40局のarchiveはそれぞれSHA検証success。失敗原因は `python scripts/d41_public_20.py train-public` の `ModuleNotFoundError: No module named 'fukaji'`、スクリプトのインポート探索パス未設定。NNUE重みの不一致・学習データ破損ではなかった。

`scripts/d41_public_20.py` にプロジェクトルートを `sys.path` へ追加する修正を[commit `8fc50e337751f018262b19295883149b66fe824d`](https://github.com/mtk-ctrl/shogi-fukaji/commit/8fc50e337751f018262b19295883149b66fe824d)で反映。研究入力JSONの意味・同40局・親を変えず、[commit `085e58c41b82a4a9e1b5076d412e1c936fc3b4ea`](https://github.com/mtk-ctrl/shogi-fukaji/commit/085e58c41b82a4a9e1b5076d412e1c936fc3b4ea)で新しいrunを起動。

修正後GitHub Actions [run **42120367483**](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42120367483)、2026-11-05 22:18〜22:19 UTCは**全ステップsuccess**。GitHubの実行ログで下記を確認。

| 指標 | 値 |
|---|---|
| 学習前モデル | D37 `170d6c1e…` |
| 学習後モデルSHA256 | `5cb31550fd9a011cace5b007df1c58c4e1889655e0a2f170869ce8eddb189e73` |
| optimizer | **Adam 2106 → 2144** |
| 新規自己対局 | 40局/40局、未決着0 |
| 学習に使用したゲーム | 新40局＋既存trainから選択40局＝80局 |
| サンプリング局面 | **2,398** |
| 学習 | 2 epoch、lr=0.001、結果75%＋TD25%・TD lambda=0.8 |
| 学習loss（epoch 0 → 2） | 0.5426779 → **0.4489994** |
| NNUE棋力向上 | **未検証** |
| 現行manifest | **D37のまま、更新なし** |

- [公開Release `fukaji-d41-d37-trained-42120367483`](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d41-d37-trained-42120367483)に、学習後モデル・optimizer・原棋譜・結果のパッケージ `D41-selfplay-training-unadopted.tar.gz`を保存。Release SHA256 **`0ceb5bc8f419579be064c97b720dde82f21fce267cc9b1bac6edb88438370731`**。Actionsの `gh release download` による再取得SHA一致を実証。
- これは**追加学習が実行でき、損失が下がった結果**であり、NNUEの棋力が高まったと断定する根拠ではない。

## 次工程（未実施）

1. **学習前D37 vs 学習後`5cb31550…`を直接比較**。F20の通常条件＝各手200ms、陣営ごとの長考1秒最大10回、同一探索/CPU/実測時刻。独立50序盤×先後2局＝**100局**、可能な限り**20独立GitHub Actions runner**で分割。局数・モデルSHA・終了条件・評価序盤除外を起動前固定。
2. 全100局の棋譜・勝敗・厳密引分・未決着・合法性・時間超過・ペア単位の不確実性を監査し、GitHub Releaseへ全原本とreceiptを保全。
3. **学習後の棋力が改善したと確認できれば現行モデルを原子的に新重みへ切替え**、次周の自己対局はその重みから始める。改善不明/悪化なら現行D37を保持。勝敗比較前に `formal/model.json` を変えない。
4. GitHubへ未移行の旧Drive資産がある場合は移行状態を別途追跡。移行完了まで旧成果は消さない。作業AIは新規Driveへの二重保存を行わない。

## 誤認防止

- D41は**20ランナー対局と追加学習を完了**。強さの実証は**未実施**。
- 単一writerで学習した新モデル`5cb31550…`は「学習後モデル」であって「現行モデル」ではない。現行はD37。
- GitHub Actionsの成功、loss低下、重み更新量は棋力向上の証明ではない。
- 第1学習runは失敗、第2学習runは成功。失敗runを消したり、最初から成功したと扱ったりしない。

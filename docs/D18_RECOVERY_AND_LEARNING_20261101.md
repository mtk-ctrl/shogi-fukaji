# D18：Work障害からの復旧と継続自己学習

2026-11-01。対象は深路だけ。共通Driveルーター→深路AGENTS→F00/F01/F10/F20/F30、現行D17、正式エンジン仕様を確認した。

## 正本から確認した引継ぎ地点

GitHub main `69e8102d40d3d8133e2e8b967ce235e5eb02b9e8` にD17実装、専用Rule、初期化結果と4周目までのレポートが存在する。learning pipeline run `41247956112` の完了・成功もAPIで再確認した。

一方、5周目、MC対TD混合8局、学習モデル一式はmainのtreeに無かった。前Workの残存フォルダから5,258,294バイトの完全状態を回収した。SHA256 `d100b8a69d86d19ac0e16976160e4af5edc3f482addb0cbf4b3d98d564d95eb3`。対応するGit blob `07850465de5b6a3fa772d5c009841c7d207a80eb` は存在したが、mainのパスには接続されていなかった。blobの保存だけで再開可能な保存の完了とはしない。

本修正ではそのarchive、5周目・8局比較のレポート、検査記録をmainのパスに固定する。旧Workの未コミットフォルダを上書きせず、新しいcheckoutで復元した。

| 確認項目 | 実体 |
|---|---|
| 保存モデル | optimizerを含む7個すべて再読込成功 |
| train | bootstrap665局＋新20局＝685局、全batch SHA一致 |
| monitor | 52局、全batch SHA一致、train親と分離 |
| 最新研究モデル | `08fe3724944c5f6433c4553711b4c7a30546c3b47d9e4d1cc4b9d6737faa753d` |
| 5周目 | 新4局すべて詰み。親比較1勝1敗1分1未決着、定期比較4勝0敗1分3未決着 |
| MC対TD | 同一親・同じ8訓練棋譜・223抽出局面。50ms、4序盤先後交替。TD2勝、MC0勝、2分、4未決着 |

8局は生棋譜を独立に合法再生し終局・勝者を再確認した。優位性の確定や正式採用は行っていない。checksumの検査と合法再生は区別する。

## 今回の実装

`fukaji.online_state` にverify/restore/packを追加。全参照モデル・optimizer、棋譜とreportのchecksum、train/monitor分離を検査する。復元は別ディレクトリで検査後に公開し、既存stateを上書きしない。packは再展開・検査まで行ってreceiptを残す。入力archiveのchecksum、パス、リンク、ファイル種別、展開上限を検査する。

再開例（リポジトリのrequirementsを導入済みのPython）：

```sh
python -m fukaji.online_state restore --root build/d17-state --archive research/d17-fast-self-learning/verified-research-state.tar.gz --sha256 d100b8a69d86d19ac0e16976160e4af5edc3f482addb0cbf4b3d98d564d95eb3
OPENBLAS_NUM_THREADS=1 python -m fukaji.online_cycle cycle --root build/d17-state --cycles 3 --periodic-every 1
python -m fukaji.online_state pack --root build/d17-state --archive build/continued-state.tar.gz
```

## 継続実験

復元した最新TD研究モデルから、最大3周の追加学習を開始。1周は新4局、親比較4局、固定anchor/過去モデル比較8局。50ms、手数上限256、結果75%＋TD25%、既存optimizer継承。各周で定期比較し、既存ゲートがpause/rollbackならそこで停止する。これは研究更新であり、正式エンジン・G8・雲路を更新しない。

## 未完了の区別

D9/PR132、D10途中の研究は保全したまま。本件で完了へ変更しない。MC対TDの8局は少数の開発比較であり、独立な方式比較の追加が必要。NNUEの玉相対特徴・幅拡張、静止探索、入玉判定、C++差分状態、D17 workflowの実dispatch確認は未完了。

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

3周を完走。計432.70秒、新12局（詰み10・千日手1・未決着1）、重複除去後10局の新しい決着経験を旧経験と混ぜ、延べ581局面を各2epoch追加学習した。未決着1局は教師にしなかった。比較は各周12局・計36局。異なる候補と相手なので、これらを同一モデルの通算勝率にしない。

| 周 | 新しい一意の決着棋譜 | 親比較（勝/敗/分/未決着） | 定期比較（勝/敗/分/未決着） | optimizer step |
|---|---:|---|---|---:|
| 6 | 3 | 2/1/1/0 | 5/1/1/1 | 1268 |
| 7 | 4 | 1/1/1/1 | 1/3/4/0 | 1272 |
| 8 | 3 | 3/1/0/0 | 4/1/3/0 | 1276 |

最新研究モデルは `e63f6f7e69919655822ce2060616d7ed612dbb08238865d0043ecd78271e9c26`。8周目の固定初期anchorとの内訳は2勝0敗2分。7周目は同anchorに0勝2敗2分で、少数局の暫定更新では悪化も起きる。今回の上限3周ではゲートによる停止・巻戻しは起きなかったが、毎周改善を保証する方式とはしない。損失低下も棋力の証明にしない。

比較手の中央値は52.94〜53.64ms、最大86.46/244.95/138.34ms。50ms要求の厳密な上限は未達で、完了depth0も各周410/593/443手ある。浅い根元探索・棋譜履歴の再構築・合法列挙の費用を次の探索効率化候補として残す。時間差の主張には非計測・同条件比較を別に要する。

過去のMC/TD比較は訓練処理自体も再実行。同一親・同一8棋譜・同一seedで両モデルの全重みとAdam momentをbit単位で再現した（差0）。これは過去の学習条件の確認であり、棋力の再現ではない。

その2つの凍結モデルを、新しい8序盤・先後16局・双方100ms・手数上限256で追加比較する。旧8局の50ms条件とは分ける。`fukaji.online_compare` は全monitor親を対局前に保護し、各ペアを保存する。失敗・中断時はlearningをpausedのまま残す。比較後も研究pointerは動かさない。

## 未完了の区別

D9/PR132はAPIでもopen・未統合と再確認。古い説明値の訂正と対局は未実施のまま保全。過去の拒否対象を本件で変更していない。

D10は文書の「初回失敗後・未集計」が古かった。実run `41165682586` はprepare・audit20本・summaryすべて成功。期限付きsummary artifact `13028798369` を取得し、ZIP SHA `5940d59fd7ba676402de152259c7f8c1f566ab286e519e21912ef11d3de5686d` と内部observations SHA `d8a0aad57f314d4439426df1c458ea027efd3acb3a9b3993c466541a10c62094` を確認して、summary/observationsを `research/d18-recovery/` に保全した。240局42,758手をActions上で再生、除外発生182局・着手差10局面、除外ONだけ/ OFFだけが相手の1手詰みを許す例は双方0。固定深さ2の診断であり同時間対局ではなく、正式採用・NNUE教師化はしていない。これら240局の独立ローカル再生は本件では行っていない。D10の強弱判断・採用は依然未完了だが、計測・集計自体の再実行は不要。

MC対TDの8局は少数の開発比較であり、独立な方式比較の追加が必要。NNUEの玉相対特徴・幅拡張、静止探索、入玉判定、C++差分状態、D17 workflowの実dispatch確認は未完了。

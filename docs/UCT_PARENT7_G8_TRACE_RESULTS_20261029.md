# 親7・凍結G8モデルの時計診断：実測・独立監査（2026-10-29）

状態：**研究結果のみ／正式採用なし**。雲路main・深路正式USI・G8モデル・既定値を変更していない。

## 目的と出所
[前回の非計測12親ABBA](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40753877836)では、探索木解放予約ONの親7（50手目）が外部206.721ms、内部205.128ms、185反復。G8モデルなし・計測付きの[PR#81](https://github.com/mtk-ctrl/shogi-fukaji/pull/81)では直接再現できなかった。

今回の[入口PR#82](https://github.com/mtk-ctrl/shogi-fukaji/pull/82)はbranch CI [40769695595](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40769695595) **379/379成功**→squash研究main `a63919ac66f96293c50196328b44993801c3bd25`→main CI [40769936392](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40769936392) **379/379成功**。研究request初期OFF、重いjob skip。

所有27repoの直近最大100run、100件超の8repoについてqueued/in_progress/pending/waiting/requestedの5状態を調査し、最終的にactive0・同目的の起動run/Issue重複なしを確認。他run停止なし。別明示request `d5349b2a386867d3a29e2605df8ee580d378692e` から1runner [実run40770234517](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40770234517) **completed/success**。

## 固定条件・実施内容
G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、240局原本SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf`を実ファイル照合。Actionsで全240局の棋譜を合法再生し、12独立親から50手目・親7（pair7）のSFENを照合した。独立コンテナでの240局全手再合法再生はpython-shogi未導入のため未実施。

同一親7・seed `20279035`・200ms・G8相対評価25%・単一合法手走査/総時計/線形根選択/解放予約をすべてON固定。**段階時計あり4回／なし4回**を交互に8呼出し。これは予約ON/OFFの比較ではない。実対局・教師生成・学習なし。

| 指標 | 計測あり4回 | 計測なし4回 |
|---|---:|---:|
| 外部呼出し時間中央値 | 194.441ms | 194.635ms |
| 外部200ms超過 | 0/4 | 0/4 |
| UCT反復合計 | 383 | 388 |
| 反復範囲 | 91〜99 | 88〜103 |

段階時計あり4回の最長単一反復は**3.010〜4.997ms**、生成Nodeは**6,497〜6,949**、最終反復終了時の残時間は**6.356〜9.984ms**。8呼出しの壁時計−process CPU時間差は中央値**0.0185ms**、最大**0.112ms**（小さな負値は計測差・丸めを含む）。選択手は`7e8e`が6回、`5f5e`が2回。すべて合法・盤面履歴復元一致。

## 独立監査・保存
artifact **12851103282**、ZIP **2,035B**、SHA256 `8738908f9ac36f2cbbe5d51b6233e41cbc18aa3cd584534891beb9df24f6ff6d`、期限**2026-11-27**。内部8行 `parent7_g8_observations.jsonl` SHA256 `e93b8fb40dea27e686dcdc3b4ac86c9ee169b28bcf50eaad29b1fc6b6277bebc`。独立取得したZIPの再計算、8行/順序/同一親・seed/4+4/中央値/超過件数/反復合計がsummaryと一致。実runのテスト3件+26件も成功。

requestはmain commit `d4e07e27e6007c11c6d073f8d6bc13a236c3e0db`で無効化。後続runのskip確認は別途記録する。

## 解釈の境界
今回の8呼出しで**元の206.721ms事象は再現しなかった**。このrunではprocess CPUと壁時計の差が小さかったが、元の事象でOS割込みがなかったとは証明できない。元のG8非計測185反復と今回の88〜103反復は、実行環境・計測・探索経路が異なるため速度改善/悪化の直接比較に使わない。計測器あり/なしも厳密な同条件性能比較ではない。

単一反復の突発的長時間化・OSスケジューラ停止・GC/解放・開始終了境界の寄与を分離するには、複数実行環境・より多くの独立観測と、計測負荷の管理が必要。今回の8回だけで200ms硬上限、予約係数最適化、棋力向上を主張しない。未決着を勝ち/引分教師にせず、NNUE PR#62の安全拒否を迂回しない。Drive原本ZIP長期保存はBLOCKED_FILE_REFERENCE等の正規解消まで保留。

漫画素材（事実に基づく案）：『同じ局面にG8を載せ、時計を二つ並べた。今回は越えなかった。でも一度越えた事実は消えない。』

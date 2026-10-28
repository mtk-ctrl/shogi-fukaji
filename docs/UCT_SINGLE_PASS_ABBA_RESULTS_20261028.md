# 深路G8：UCT合法手リスト単一走査ABBA実測結果（2026-10-28）

## 結論
研究専用・既定OFFの単一走査は、**同じ12局面・同じ200ms・同じG8モデル・同じ1runner**で、UCT反復合計を旧3,223回から新5,630回へ **+74.68%** 増やした。一方、実時間中央値はOFF 199.216ms / ON 199.447msで新方式が0.231ms長く、200ms超過はOFF 7/24→ON 6/24で残った。12親局面の2回平均ではONが速いのは6/12のみ。探索処理量の増加は観測できたが、**厳密200ms遵守、棋力向上、正式採用は証明されていない**。

## 実施経路
- [PR#67](https://github.com/mtk-ctrl/shogi-fukaji/pull/67)の単一走査は既定OFF、研究main `1ee46bb778373ec851b12c2450ee56f5f43da89d`。main CI [40699295110](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40699295110) 成功。
- [PR#68](https://github.com/mtk-ctrl/shogi-fukaji/pull/68)で、専用 `fukaji/uct_single_pass_abba.py`、検査、既定OFFのrequestと1runner workflow/README同一commit、固定計画を追加。branch CI [40702772250](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40702772250) 338/338成功。squash研究main `ca47edc0cfb63abb38c272089a183bdb485b5867`、main CI [40703134064](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40703134064) 成功、専用既定OFF入口のskip CI [40703131871](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40703131871) 成功。
- 27repo最近100runと、100超8repoのqueued/in_progress/pending/waiting/requestedを個別確認してactive0、同目的重複なし。別明示request commit `c4ecb738d8811f91699372fb9948bd6cc7f14604`から**1runner**を起動。
- 実run [40703556000](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40703556000) completed success。原本240局を全手合法再生。既定OFFへ復元commit `3353f1d41497c4f1f097daf4211974f6af3de845`、[skip run40703734515](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40703734515)で重い工程全skipped確認。他run停止なし。

## 凍結入力と独立監査
- G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、モデルrun40636602223。
- G8/G7原本240局JSONL SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf`、原本run40637486947。
- 12/50/110手目各4局面、親12組独立・SFEN同一順序、各方式24観測・計48行、OFF→ON→ON→OFF、seed固定、非計測UCT、各200ms、深さ上限12。
- artifact ID `12822572871`、ZIP 2,729 bytes、SHA256 `23f5276aad74c168a0481ed5f0c27a93027b858638e1f6d51e5f1da905f9b9d3`。GitHub公表digestと独立取得バイトのSHA一致。
- 内部48行 `abba_observations.jsonl` SHA256 `f1dc392895875f3a8819ac50b28309ca73f531e1bdabfb2982e45d3bdbfcf3a3`。独立に全48行・12親・ABBA同一局面・時間/反復/統計を再集計一致。
- [Google Drive原本ZIP](https://drive.google.com/file/d/1-35o9DZmadjuMt12QMBQiapvEwZOOzGg/view)に保存し、再取得したZIPがGitHub原本とバイト単位で一致。30日GitHub artifact期限は2026-11-27。

## 比較値
| 指標 | OFF（二重走査） | ON（単一走査） |
|---|---:|---:|
| 観測 | 24 | 24 |
| UCT反復合計 | 3,223 | **5,630（+74.68%）** |
| 思考時間中央値 | 199.216ms | 199.447ms |
| 思考時間平均 | 197.356ms | 195.733ms |
| 最大時間 | 206.512ms | 206.117ms |
| 200ms超過 | 7/24 | 6/24 |
| 2回平均でONが速い親局面 | — | 6/12 |

手数段階別の反復合計は12手目OFF1,887/ON3,228、50手目OFF1,050/ON1,696、110手目OFF286/ON706。これらは異なる局面の合計であり、棋力差ではない。旧方式と新方式で着手が変わる局面があっても、優劣の証明にはならない。

## 解釈と次の課題
- 二重走査除去でUCTの反復処理量が増えるという仮説は同一局面の実測で支持された。ただし反復数は探索節点数や強さと同一ではない。
- 思考時間の中央値は改善しておらず、200ms超過も残る。時間予約の上限を厳密保証する仕組みではない。1反復の遅延分布と合法手生成の最悪ケースを別に調べる。
- 同一12局面・各2回は標本が小さく、OSスケジューラの影響を排除できない。強さを示すには研究候補を正式採用せず、先後交替の独立対局を別途設計・検証する。
- NNUE学習PR#62は安全拒否を迂回せず未統合・未学習。雲路main・深路既定USI/モデルは不変更。未決着を勝ち・引分の教師にしていない。

## 漫画素材
「98%を占める箱を開けると、合法手を二度数えていた。1回に直すと探索は74.7%多く進んだ。しかし時計はまだ200msを越える。速さと強さ、速さと締切は別の課題だった。」

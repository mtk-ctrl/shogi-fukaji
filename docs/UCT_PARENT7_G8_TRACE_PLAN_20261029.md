# 親7・凍結G8モデルの少数時計診断計画（2026-10-29）

状態：研究専用、既定OFF、正式採用なし。雲路main・深路正式USI・G8モデル・既定値を変更しない。

## 問題
非計測の固定12親ABBA [run40753877836](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40753877836)で、予約ONの50手目親7が外206.721ms、内205.128ms、185反復。ZIP SHA256 `3118b982bf2f8a60fb756fbe2c87bb77980ee3cd83bae16f62c4f0f3dc165873`。前回の[PR#81軽量診断](https://github.com/mtk-ctrl/shogi-fukaji/blob/main/docs/UCT_PARENT7_LIGHT_TRACE_20261029.md)はmodel=Noneかつ計測付きであり、G8非計測の直接再現ではない。

## 固定条件と入口
専用request `.github/fukaji-uct-parent7-g8-trace-request.json` は初期 `enabled:false`。明示的な別commitでのみ有効化し、実測後disabledへ戻す。1runner・同一親7・同seed `20279035`・200ms・G8相対評価25%・合法手単一走査ON・総時計ON・線形根選択ON・探索木解放予約ONを固定。計測付き4回と計測なし4回を順序交互に計8回。**比較するのは計測負荷の観察であり、予約ON/OFFのABBAではない。**

G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、240局原本SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf`を実バイト照合し、全240局を合法再生して12独立親を復元。親7のSFENと50手目・pair7を厳格照合。

各観測で合法手・盤面履歴復元・反復数・呼出し外部/内部時計・process CPU時間を記録。計測付きは根準備・探索ループ・最長反復・最終反復開始/終了残時間・生成Node・盤面復元確認・根最終手選択・内部超過を別記する。8行JSONLとsummaryのSHAを期限付きartifactに保存。

## 解釈と安全
- process CPU時間と壁時計の差はスケジューラ停止の候補を示すだけで、OS割込みの直接証明ではない。
- 計測器が実行順序・負荷を変えるため、計測あり/なしの200msや反復数を同条件の性能比較とみなさない。8呼出しで厳密な200ms保証、係数最適化、棋力向上を主張しない。
- 長い1反復の実測が得られても、G8元の206.721ms事象そのものを再現したと断定しない。
- 反復0でも合法手なら正当記録。未決着は勝ち/引分の教師にしない。新対局・学習・正式採用なし。
- PR→branch CI→squash研究main→main CIの後、所有27repoのqueued/in_progress/pending/waiting/requestedと同目的重複を調べ、空きなら別明示requestで1runner起動。成果物のSHA・8行/合法性/統計を独立監査し、request disabled/skipまで確認。他runは停止しない。
- Driveへの原本ZIP保存で以前生じたBLOCKED_FILE_REFERENCEは迂回しない。

漫画素材：『同じ局面を、今度は本物のG8評価を載せて測る。CPUの時計と壁の時計がずれたときも、理由を決めつけずに記録する。』

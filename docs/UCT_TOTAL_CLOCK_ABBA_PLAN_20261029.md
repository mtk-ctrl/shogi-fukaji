# 深路UCT総時間予算ABBA実測入口（2026-10-29、研究候補）

## 比較条件
凍結G8モデルSHA `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、240局原本SHA `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf`。240局の全手合法再生後、12/50/110手目各4親の計12局面を同一seed・同一1runner・各200msで比較。単一走査ON固定、総時間予算のみOFF→ON→ON→OFF、計48観測。局面評価モデルと探索深さ12は固定。

## 安全と監査
専用requestは既定 `enabled:false`、main限定、scheduleなし、最小読取権限、cancel-in-progress:false。PRブランチCI→squash研究main→main CI→27repoの5状態/同目的重複→別commitの明示request→実run→SHA/合法性/統計監査→request disabled/skip。実run前にモデル・原本SHAを再照合。根準備が予算を使い切った場合、反復0でも合法手なら有効な観測とし、未決着勝ち・引分ラベルを付けない。外部呼出時間と内部探索計時は異なる境界を持つため両方保存する。200ms硬い上限保証・棋力向上・正式採用は主張しない。原本ZIPは期限付きartifactからDriveへ保存・再取得SHA検査。

## 物語素材
『速い探索は作れた。次に問われたのは、時計がいつ動き始めるかだった。二度数える無駄をなくしても、準備に使う時間は消えない。』

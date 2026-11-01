# D27：別環境から復旧できる保存点と実行継続

2026-11-02。対象は深路のみ。正式USI/配布モデルの変更なし。

## 確認した障害

D26の新400局を元Workで起動した後、Automationsは別containerで起動して元workdir/exec/lockを読めず、結果を取得しなかった。自動確認の実run c7997352は記録上完了だが、研究の400局完了を意味しない。本人の継続依頼で元Workを調べてもevaluation worktreeは消失、全scratch探索で該当progress/reportなし、元exec18484はUnknown process idだった。

本人限定Driveには第15周完了＋400局開始前の全状態が残った。SHA `71f309bf49f42f85a8cfca65171221e4eb5220438c0116adbc7b9a028ecadf85`、29models/train753/monitor892/15cycles/optimizer1324。今回再取得・SHA照合・index実復元を行い、候補ae53ce0d、比較親14周2161b2ba/初回5cdfebea、全100序盤、controller/engine SHA、CPU条件が元planと一致することを確認した。

旧試行の未保存対局は完了結果として数えず、同じ凍結planを0/100組から復旧した。これは実行環境喪失への復旧で、勝敗に応じた再測定・有利な試行の選別ではない。モデルの再初期化・再学習はしていない。既存D24/D25の完成結果と第15周モデル/optimizerは保持している。

## 修正した仕組み

`fukaji.portable_resume.create_index` は、外部アップロードを行わず、実際のconnector観測を受け取って次を検証する。

- ownerのみ・shared=falseのremote metadata、archiveの再取得SHA一致。
- 既存archive検証器による安全な実復元、モデル/optimizer/親分割/plan/reportの整合。
- `recovery_checkpoint_only`：実行中の途中保存。別環境で元writerを確認せず再開する許可ではない。
- `stopped_handoff`：元rootのexclusive lockを取得でき、archiveのstateと現stateが一致する停止境界。active writer中/古いarchiveは拒否する。
- 再開方法は永続archive・exact code commit・plan/source SHA・復元後のcommand。元session IDは可搬なhandleとして扱わない。

F01/F30にも同じ境界を反映した。ローカル長時間計算は同じWorkで完了と保存まで追い、20序盤ごとの不変checkpointを本人限定Driveへ保存する。別環境に読める成果物がない状態で、ローカルrun終了を待つ予約だけを根拠に継続成功とは報告しない。

## 検証

active lock下でstopped_handoffを拒否しつつ復旧checkpointは作れること、古いarchive/公開または不明権限/readback不一致を拒否すること、paused/pendingのplanを別directoryへindex実復元できることを確認した。D26の凍結controller/engineは変更していない。復旧した400局の結果はD26へ記録する。

## 実際の復旧完了

同じWorkで100序盤組400局を終え、全68,143手の合法再生と終局一致を確認。第15周は第14周に104勝93敗3分、初回学習済みに89勝106敗5未決着。事前判定はいずれも未確定。実行1752.438秒。20/40/60/80組のcheckpointはそれぞれ本人限定Driveへ保存し、ownerのみmetadata・再取得SHA・実復元・portable indexを検証した。

今回の再発防止は、読めないローカルprocessを予約へ渡さず、実際に可搬な保存点を作ること。自動確認環境での研究継続全体が修復済みとは主張しない。別directoryへの実復元と元writer停止境界の検証までを確認した。

D28の同一NNUE統合32局も同じWorkで完了し、全5710手を合法再生。終わったローカルrunを別環境の予約へ渡さず、両処理が停止した境界の全状態を最終保存する。標準の再開indexで別directoryへ実復元し、結果/モデル/optimizer/未完了writerの有無を確認する。

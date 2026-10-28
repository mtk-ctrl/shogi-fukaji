# 深路G8：UCT単一走査の同一12局面ABBA実測計画（2026-10-28）

## 目的と前提
PR#67で研究専用・既定OFFの `single_pass_legal_moves` を追加し、研究main `1ee46bb778373ec851b12c2450ee56f5f43da89d` に統合した。main push CI `40699295110` はverify成功。27repoの最近100run、100超8repoのqueued/in_progress/pending/waiting/requestedは2026-10-28 23時台の確認で全0。ただし起動直前に再照合する。

今回の比較はpython-shogiの `list(board.legal_moves)` と内包表記 `[m for m in board.legal_moves]` の違いだけで、探索戦略・モデル・合法性・既定値を変更しない。

## 凍結入力
- G8モデルSHA256：`18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492`、run40636602223。
- G8/G7独立240局原本JSONL SHA256：`60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf`、run40637486947。
- 原本240局全手合法再生、先後ペア、終局・未決着の扱いを検査。12局面は12/50/110手目各4、親ペア重複なし、SFEN重複なし。
- 同一1runner、同一モデル、同一seed、200ms、UCT深さ12、非計測、OFF→ON→ON→OFF。各方式24観測、合計48行。

## 実行ゲート
1. branch PR CIのテスト成功を確認し、研究用mainへsquash統合。main push CI成功を直接確認する。
2. 27repoの5状態と同目的の起動/待機runを再調査し、他の有効runを停止しない。
3. 別commitの明示requestだけで実測。YAMLはmain限定・read-only・1runner・scheduleなし・既定OFF。
4. 48行、親12、合法手・盤面復元、固定モデル/原本SHA、観測時間/反復数/200ms超過を検証。成果物ZIPと内部JSONLのSHA256を独立取得・再計算する。
5. requestをdisabledへ戻し、後続の重いjobがskipであることを確認。GitHub技術正本・Drive原本・Addnessへ記録。

## 解釈の限界
同一runnerのABBAでもOSスケジューラ揺らぎは消えない。各方式24観測は記述統計であり、200ms厳密保証・棋力向上・正式採用の証明ではない。局面差による選択手の違いを棋力差とみなさず、未決着を勝ち/引分の教師にしない。雲路main・深路既定USI/モデルを変更しない。NNUE学習PR#62の安全拒否書込みを再試行・迂回しない。

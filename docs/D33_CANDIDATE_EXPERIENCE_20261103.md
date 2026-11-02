# D33：候補同士の新経験を次の追加学習へつなぐ

2026-11-03。D31の400局評価が進んでいる間、別のCPUで次の学習用TRAINを生成する。D31の代表保持候補c8ad43bfと同seed対照b73f0c24（ともにAdam1580）を凍結し、新しい32開始親の先後64局を収集する。

D31評価の全100親は結果を見る前に保護済み。過去のTRAIN/monitorを含む除外一覧から新しい親を選び、seed202611033300、C++基本NNUE/q0/50ms/追加256手、初期24手の探索確率5%とする。役割は開始前からTRAIN。未決着は保持するが教師にせず、対局の勝敗を強化比較へ加算しない。

D31のworker CPU0〜3/writer4と分け、D33はCPU5・6/writer7を使う。評価root/state/既存モデルには書き込まず、別出力bundleだけを単一writerで更新する。元2モデルの不変snapshot・由来を同梱し、出力bundle単体で全64局を合法再生できる。原対局・序盤親・モデル・seed・条件・SHA・終局根拠を検証する。

実装 `fukaji/followup_train_collection.py`。完了後、bundleを本人限定Driveへ保存し、メタデータ・再取得SHA・別領域への展開と全件再検証を行う。D31 rootのcheckpointと、別途進行中のD33 bundleを混在させない。D33 bundleをD31の進行中archiveへコピーしない。

次の追加学習はD31の全結果を確認してから、起点/凍結教師/配分/更新量を固定する。D31代表seed/条件/終点を変更せず、正式v0.2.0も固定する。

Goal: https://www.addness.com/goals/e853f54c-b919-44d8-96e5-14c7f9c999c3

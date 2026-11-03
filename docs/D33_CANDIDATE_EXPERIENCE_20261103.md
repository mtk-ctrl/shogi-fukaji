# D33：候補同士の新経験を次の追加学習へつなぐ

2026-11-03。D31の400局評価が進んでいる間、別のCPUで次の学習用TRAINを生成する。D31の代表保持候補c8ad43bfと同seed対照b73f0c24（ともにAdam1580）を凍結し、新しい32開始親の先後64局を収集する。

D31評価の全100親は結果を見る前に保護済み。過去のTRAIN/monitorを含む除外一覧から新しい親を選び、seed202611033300、C++基本NNUE/q0/50ms/追加256手、初期24手の探索確率5%とする。役割は開始前からTRAIN。未決着は保持するが教師にせず、対局の勝敗を強化比較へ加算しない。

D31のworker CPU0〜3/writer4と分け、D33はCPU5・6/writer7を使う。評価root/state/既存モデルには書き込まず、別出力bundleだけを単一writerで更新する。元2モデルの不変snapshot・由来を同梱し、出力bundle単体で全64局を合法再生できる。原対局・序盤親・モデル・seed・条件・SHA・終局根拠を検証する。

実装 `fukaji/followup_train_collection.py`。完了後、bundleを本人限定Driveへ保存し、メタデータ・再取得SHA・別領域への展開と全件再検証を行う。D31 rootのcheckpointと、別途進行中のD33 bundleを混在させない。D33 bundleをD31の進行中archiveへコピーしない。

次の追加学習はD31の全結果を確認してから、起点/凍結教師/配分/更新量を固定する。D31代表seed/条件/終点を変更せず、正式v0.2.0も固定する。

Goal: https://www.addness.com/goals/e853f54c-b919-44d8-96e5-14c7f9c999c3

## 次の追加学習（D31全結果後に実行する事前条件）

D31を全400局完了し、保持候補c8ad43bfが研究用起点へ選ばれた場合だけ、そのAdam1580から3seedを継続する。選ばれていなければこの学習器は拒否し、全結果から研究方針を見直す。D31結果の途中で学習を始めない。

D33の46決着（43詰み＋3通常千日手）・D31の54・D30の55・bootstrap665・第1〜15周80を重複除去して保持。凍結親c8の結果75%＋TD25%/λ.8教師を全cohortで作り直す。各棋譜最大32抽出の共通poolから8192位置を復元抽出し、最新2048・直前D31を1024・その前D30を1024・bootstrap2048・第1〜15周2048を学習する。第15周は古いNNUE経験として保持する。

2epoch/batch128/lr.001/128追加stepでAdam1708。seed202611033301/202611033302/202611033303を全件保存し、対局代表は先頭33301へ固定する。研究用latestと正式モデルは学習器自身では変更しない。実装 `fukaji/followup_learning.py`。

追加学習後は代表対学習直前親c8を新100序盤×先後=200局、C++q0/100ms/総512手、評価seed202611033380で測る。1比較の保守的標本誤差幅約±14点を予算の目的とし、20組ずつ停止保存して全終点へ進む。未決着を別表示し、途中成績で停止・seed変更・正式採用をしない。

## 現在の保存状態と再開条件

全3候補の追加学習と、D31全400局・D33全TRAIN・次200局planを含む完全状態を保存済み。事前代表e0cbd2abe53b1d151bb73407f848b4087456c1466121333ad490a96aee131b0a/Adam1708を、親c8ad43bf/Adam1580と比較する。

[全状態](https://drive.google.com/file/d/1l-ukhYLegviNJ-Qt9Wz4rmJtijWboiP5/view)は66,725,804bytes、SHA256 `3f70f9c720d5f360753b85bf8941cf12cb75b3f7f51c9972c6a02f6a15dd1762`。[再開index](https://drive.google.com/file/d/1YGAMrWE31sv2znT7ufY_UHEeMzAcV0IC/view)でowner-only・遠隔SHA・標準復元・元state一致を検証済み。47models/train945/monitor2524/15cycles、研究latestはc8。D33候補をlatestや正式版へ反映したという意味ではない。

保存直後の200局起動要求は `exec-server transport disconnected; recovery timed out after 25s` で拒否され、実行環境がunavailableとなった。保存済み進捗は0/100組。計算中との報告や、実処理のないタイマー予約をしない。

再開時は元環境のwriter・progress・lockを確認し、稼働済みなら二重起動しない。source `31e799a9a506befad5bd8b5ce8ccab26edc33863` とindex照合で標準復元する。依存は今回checkoutの `build/runtime-deps` へrequirements.txtどおり導入し、`run-research.sh` で検査して `./run-research.sh -m fukaji.frozen_evaluation run --root RESTORED_ROOT --name d33-followup200` を実行する。source/engine/model/Adam/CPUをplanと照合し、既存の完了組を保持、20組ずつ保存・復元して全100組へ進む。


## 後続判断：代表NNUEの正式採用

2026-11-03、オーナーの明示指示により代表 `e0cbd2abe53b1d151bb73407f848b4087456c1466121333ad490a96aee131b0a` をFUKAJI v0.3.0へ正式採用した。D33の固定200局は採用時点で0/200局のため、棋力向上が実証されたという意味ではない。正式仕様と配布状況は [D34](D34_FORMAL_NNUE_RELEASE_20261103.md)、判断理由は [採用記録](decisions/2026-11-03_D33_NNUE正式採用.md) に記録する。研究latestはc8ad43bfのまま、元の比較は続行する。

# 深路：NNUE高速自己強化 — 現在地

2026-11-03更新。現行状態は以下。研究方針は[D17](D17_FAST_SELF_LEARNING_20261031.md)、判断正本は[decisions/](decisions/)、各比較結果はD系文書。ChatGPT・Claudeで同一Rule、Addness保留。

## 現在の正式版
**FUKAJI v0.3.0（D33 NNUE e0cbd2ab）**。本人判断で正式採用。[D34](D34_FORMAL_NNUE_RELEASE_20261103.md)・[正式採用判断](decisions/2026-11-03_D33_NNUE正式採用.md)・[現行仕様](現行エンジン仕様.md)を参照。採用時点ではD33比較0局。後日完了した結果を採用時点の根拠に遡及しない。旧v0.2.0は[D32](D32_FORMAL_NNUE_RELEASE_20261102.md)に保全。

## 現在の研究
- [D31](D31_TWO_BATCH_RETENTION_20261103.md)：400局/65,551手完了。c8対当時の正式NNUEは119勝76敗1分4未決着、対新経験のみ候補は97勝97敗4分2未決着。保持方式の候補間優位は未確定。
- [D33](D33_CANDIDATE_EXPERIENCE_20261103.md)：TRAIN64局（46決着）、Adam1580→1708の3seed追加学習。固定100序盤×先後**200局完了**。e0対親c8：113勝83敗1分3未決着、34,726手合法再生。95%区間43.17〜71.83%、50%を含み棋力差未確定。旧「起動拒否・待機」は履歴。
- [D35](D35_SELFPLAY_NNUE_CONTINUATION_20261103.md)：正式e0の学習状態から別領域で第16・17周自己対局・追加学習。研究latest db94693a/Adam1722。固定82序盤×先後164局：84勝72敗5分3未決着、31,018手合法再生。95%区間が50%を含み棋力差未確定。
- [D36](D36_SELFPLAY_NNUE_64_GAMES_20261103.md)は新32序盤・自己対局64局（56詰み、8未決着）を2回の追加学習につないだ。研究latest `7d74b86b`／Adam 1850。新20序盤を使った50ms比較で、学習前 `db94693a` に24勝16敗、正式 `e0cbd2ab` に22勝17敗1分。次の判断は[研究記録](decisions/2026-11-03_D36_研究用NNUE継続.md)。
- 最新の**保存済み再開点**は[D36完全状態](https://drive.google.com/file/d/1zhJnQ9krUjbcvTm0hXQou1Vnb2zAzL2G/view)と[停止済みindex](https://drive.google.com/file/d/1b4c70boQyHHqivukDbZ7QCJBnO9sRZGn/view)。本人限定権限、再取得SHA、別領域実復元、両学習結果と比較結果の一致を確認。以前のD35再開点は[D35](D35_SELFPLAY_NNUE_CONTINUATION_20261103.md)に保全。

## 実行・判断の境界
最新main commit、稼働Actions、他AI/元writer、正式manifest、凍結plan、復旧archiveのSHAを確認。D17の結果重視学習とデータ分離を維持。NNUEは現行正式版から直接学習し、学習前snapshotとの直接対局を通過した重みを新正式版として次周へ継続する（2026-11-04決定、F00/F10/F20参照）。旧来の別研究候補と手動昇格の流れを次回学習に使わない。現行workflowが未対応なら先に実装を合わせる。少数局の故障確認は棋力実証と呼ばない。

## 過去の結果へのリンク
[D17](D17_FAST_SELF_LEARNING_20261031.md) / [D18](D18_RECOVERY_AND_LEARNING_20261101.md) / [D20](D20_RELIABLE_RESUME_AND_TARGET_TRIAL_20261101.md) / [D21](D21_KING_RELATIVE_NNUE_20261101.md) / [D22](D22_NNUE_QUIESCENCE_20261101.md) / [D23](D23_NATIVE_NNUE_SEARCH_20261101.md) / [D24](D24_FROZEN_NNUE_EVALUATION_20261101.md) / [D25](D25_REPLAY_RATIO_20261101.md) / [D26](D26_EXPERIENCE_DIVERSITY_20261101.md) / [D27](D27_PORTABLE_RECOVERY_20261102.md) / [D28](D28_NATIVE_ACCUMULATOR_20261102.md) / [D29](D29_STRATIFIED_REPLAY_20261102.md) / [D30](D30_FRESH_EXPERIENCE_20261102.md) / [D31](D31_TWO_BATCH_RETENTION_20261103.md) / [D32](D32_FORMAL_NNUE_RELEASE_20261102.md) / [D33](D33_CANDIDATE_EXPERIENCE_20261103.md) / [D34](D34_FORMAL_NNUE_RELEASE_20261103.md) / [D35](D35_SELFPLAY_NNUE_CONTINUATION_20261103.md)。[旧ROADMAP](archive/2026-10-31_ROADMAP_before_D17.md)は履歴。古い「現在」「次」は現行指示ではない。

[F01](rules/F01_継続実行とタイマー.md) / [F30](rules/F30_実行保存と再開.md) / [Drive容量候補台帳](operations/2026-11-03_Drive容量と保持候補.md)に従う。Addness Goalは保留中・履歴参照のみ任意。

## 2026-11-04追記：D37保存成果と次の正式比較

本人限定Driveの[D37最終アーカイブ](https://drive.google.com/file/d/1XYvT7jA_0fqhPuo5v_uUIrxa4Crbocf-/view)を実取得・読込し、次の実記録を確認した。
- **現行正式版は依然v0.3.0 / D33 NNUE e0cbd2ab（Adam1708）**。D36は `7d74b86b`（Adam1850）、D37最終は `170d6c1e`（Adam2106）。D37はD36から各200局の自己対局・学習を2回行い、完全stateを保存。
- 旧条件1手100ms、長考なしの固定対局：**D36対正式e0が241勝150敗1分8未決着（400局）**、**D37対D36が129勝66敗1分4未決着（200局）**。当該分布では新モデルの勝ち越しが認められるが、本人指定の通常200ms・1秒長考10回までとは別条件。
- 2026-11-04のローカル3並列・100msでのD36対正式e0新100局比較は、26局（D36 19勝7敗）で停止。**74局未実施かつ時間・20ランナー条件不一致のため正式棋力判断に使わない。** [途中証拠](https://drive.google.com/file/d/1WvD0bvYy32jlQ8sseLcUQOjD1UA6lOTy/view)。
- D37の学習planにはrunners=20、比較planにはworkers=8と記載があるが、これは実際のGitHub Actions独立20ランナー起動証拠ではない。今回のローカル比較に20ランナーは使用していない。
- 優先する次工程：正規200ms＋長考1秒×10、独立50序盤×先後100局、20実runnerを使った**D37対正式e0の直接比較**。成功ならD37を正式NNUEとして採用し次周の親にする。結果が不確定ならD36等との追加比較を計画し、条件付きで判断する。現行正式版重みは正規比較と保存・復元が成功するまで変更しない。
- 操作・判断の正本：[2026-11-04の追加判断](decisions/2026-11-04_標準対局時間と最強NNUEへの更新.md)、[F20](rules/F20_研究更新と棋力検証.md)、[F30](rules/F30_実行保存と再開.md)。本追記は実行計画であり、正規100局比較を実行済みとは扱わない。

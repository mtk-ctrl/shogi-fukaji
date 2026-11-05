# 深路：NNUE高速自己強化 — 現在地

2026-11-04更新。現行状態は以下。研究方針は[D17](D17_FAST_SELF_LEARNING_20261031.md)、判断正本は[decisions/](decisions/)、各比較結果はD系文書。ChatGPT・Claudeで同一Rule、Addness保留。

## 現行モデル
**FUKAJI v0.4.0（D37 NNUE 170d6c1e、Adam2106）**。本人の2026-11-04指示で採用。旧e0/v0.3.0は復元用。本人限定[現行モデル配布](https://drive.google.com/file/d/1gJfz9q8TtSsyynkUrC7uP7KTwPjbNH1O/view)、[D37完全stateとoptimizer](https://drive.google.com/file/d/1XYvT7jA_0fqhPuo5v_uUIrxa4Crbocf-/view)、[判断](decisions/2026-11-04_D37を現行モデルに採用.md)、[現行仕様](現行エンジン仕様.md)。学習は**D37を親**に、更新前snapshotと更新後を直接比較し、改善なら新現行モデルへ更新する。D37対D36は100ms/q0の200局で129勝66敗1分4未決着。200ms/長考10回での強さは未測定。

## 現在の研究
- [D31](D31_TWO_BATCH_RETENTION_20261103.md)：400局/65,551手完了。c8対当時の正式NNUEは119勝76敗1分4未決着、対新経験のみ候補は97勝97敗4分2未決着。保持方式の候補間優位は未確定。
- [D33](D33_CANDIDATE_EXPERIENCE_20261103.md)：TRAIN64局（46決着）、Adam1580→1708の3seed追加学習。固定100序盤×先後**200局完了**。e0対親c8：113勝83敗1分3未決着、34,726手合法再生。95%区間43.17〜71.83%、50%を含み棋力差未確定。旧「起動拒否・待機」は履歴。
- [D35](D35_SELFPLAY_NNUE_CONTINUATION_20261103.md)：正式e0の学習状態から別領域で第16・17周自己対局・追加学習。研究latest db94693a/Adam1722。固定82序盤×先後164局：84勝72敗5分3未決着、31,018手合法再生。95%区間が50%を含み棋力差未確定。
- [D36](D36_SELFPLAY_NNUE_64_GAMES_20261103.md)は新32序盤・自己対局64局（56詰み、8未決着）を2回の追加学習につないだ。研究latest `7d74b86b`／Adam 1850。新20序盤を使った50ms比較で、学習前 `db94693a` に24勝16敗、正式 `e0cbd2ab` に22勝17敗1分。次の判断は[研究記録](decisions/2026-11-03_D36_研究用NNUE継続.md)。
- 最新の**学習再開起点**は[D37完全state](https://drive.google.com/file/d/1XYvT7jA_0fqhPuo5v_uUIrxa4Crbocf-/view)（モデル170d6c1e / Adam2106）。D36完全状態と停止indexは旧記録として保全。本人限定権限、再取得SHA、別領域実復元、両学習結果と比較結果の一致を確認。以前のD35再開点は[D35](D35_SELFPLAY_NNUE_CONTINUATION_20261103.md)に保全。

## 実行・判断の境界
最新main commit、稼働Actions、他AI/元writer、正式manifest、凍結plan、復旧archiveのSHAを確認。D17の結果重視学習とデータ分離を維持。NNUEは現行正式版から直接学習し、学習前snapshotとの直接対局を通過した重みを新正式版として次周へ継続する（2026-11-04決定、F00/F10/F20参照）。旧来の別研究候補と手動昇格の流れを次回学習に使わない。現行workflowが未対応なら先に実装を合わせる。少数局の故障確認は棋力実証と呼ばない。

## 過去の結果へのリンク
[D17](D17_FAST_SELF_LEARNING_20261031.md) / [D18](D18_RECOVERY_AND_LEARNING_20261101.md) / [D20](D20_RELIABLE_RESUME_AND_TARGET_TRIAL_20261101.md) / [D21](D21_KING_RELATIVE_NNUE_20261101.md) / [D22](D22_NNUE_QUIESCENCE_20261101.md) / [D23](D23_NATIVE_NNUE_SEARCH_20261101.md) / [D24](D24_FROZEN_NNUE_EVALUATION_20261101.md) / [D25](D25_REPLAY_RATIO_20261101.md) / [D26](D26_EXPERIENCE_DIVERSITY_20261101.md) / [D27](D27_PORTABLE_RECOVERY_20261102.md) / [D28](D28_NATIVE_ACCUMULATOR_20261102.md) / [D29](D29_STRATIFIED_REPLAY_20261102.md) / [D30](D30_FRESH_EXPERIENCE_20261102.md) / [D31](D31_TWO_BATCH_RETENTION_20261103.md) / [D32](D32_FORMAL_NNUE_RELEASE_20261102.md) / [D33](D33_CANDIDATE_EXPERIENCE_20261103.md) / [D34](D34_FORMAL_NNUE_RELEASE_20261103.md) / [D35](D35_SELFPLAY_NNUE_CONTINUATION_20261103.md)。[旧ROADMAP](archive/2026-10-31_ROADMAP_before_D17.md)は履歴。古い「現在」「次」は現行指示ではない。

[F01](rules/F01_継続実行とタイマー.md) / [F30](rules/F30_実行保存と再開.md) / [Drive容量候補台帳](operations/2026-11-03_Drive容量と保持候補.md)に従う。Addness Goalは保留中・履歴参照のみ任意。

## 2026-11-04追記：D37保存成果と次の正式比較

本人限定Driveの[D37最終アーカイブ](https://drive.google.com/file/d/1XYvT7jA_0fqhPuo5v_uUIrxa4Crbocf-/view)を実取得・読込し、次の実記録を確認した。
- **以下はD37採用前の2026-11-04時点の記録。現在はv0.4.0/D37を現行モデルとして採用済み**。当時の現行版はv0.3.0 / D33 NNUE e0cbd2ab（Adam1708）。D36は `7d74b86b`（Adam1850）、D37最終は `170d6c1e`（Adam2106）。D37はD36から各200局の自己対局・学習を2回行い、完全stateを保存。
- 旧条件1手100ms、長考なしの固定対局：**D36対正式e0が241勝150敗1分8未決着（400局）**、**D37対D36が129勝66敗1分4未決着（200局）**。当該分布では新モデルの勝ち越しが認められるが、本人指定の通常200ms・1秒長考10回までとは別条件。
- 2026-11-04のローカル3並列・100msでのD36対正式e0新100局比較は、26局（D36 19勝7敗）で停止。**74局未実施かつ時間・20ランナー条件不一致のため正式棋力判断に使わない。** [途中証拠](https://drive.google.com/file/d/1WvD0bvYy32jlQ8sseLcUQOjD1UA6lOTy/view)。
- D37の学習planにはrunners=20、比較planにはworkers=8と記載があるが、これは実際のGitHub Actions独立20ランナー起動証拠ではない。今回のローカル比較に20ランナーは使用していない。
- 旧計画：正規200ms＋長考1秒×10、独立50序盤×先後100局、20実runnerを使ったD37対e0比較。**その後本人がD37を現行モデルへ直接採用することを指示したため、当該比較は採用の前提ではなくなった**。次工程はD37からの自己対局・追加学習。結果が不確定ならD36等との追加比較を計画し、条件付きで判断する。現行正式版重みは正規比較と保存・復元が成功するまで変更しない。
- 操作・判断の正本：[2026-11-04の追加判断](decisions/2026-11-04_標準対局時間と最強NNUEへの更新.md)、[F20](rules/F20_研究更新と棋力検証.md)、[F30](rules/F30_実行保存と再開.md)。本追記は実行計画であり、正規100局比較を実行済みとは扱わない。

## 2026-11-04：D39、現行D37からの最初の自己対局・学習

- 現行モデルを**v0.4.0/D37(170d6c1e、Adam2106)**へ切り替え済み。GitHub `formal/model.json`、本人限定Drive [推論配布](https://drive.google.com/file/d/1gJfz9q8TtSsyynkUrC7uP7KTwPjbNH1O/view) についてSHA・起動・配布再読込を確認。旧e0/v0.3.0は復元用に保持。本人の[決定](decisions/2026-11-04_D37を現行モデルに採用.md)。
- D39 **新規自己対局4局/4詰み、合法再生済み**。通常1手200ms、1秒長考は各陣営10回以内（実際は各局10回）、合計424追加手。実行は**ローカル1環境・GitHub独立20runner=0**。長い局は各手のcheckpointからプロセスを分けて再開。連続1プロセスの途中ハングを再現、根本原因は未特定。
- 同4局の115局面＋許可済み旧train4局の62局面、計177局面から**D37直接追加学習**。Adam2106→2112、学習後重み`4c37e812`。損失低下のみで棋力は未確定。**現行モデルはD37のままで、学習後2112の採用は比較待ち**。
- [D39詳細](D39_CURRENT_D37_SELFPLAY_REINFORCEMENT_20261104.md)、[本人限定Drive D39完全初回checkpoint](https://drive.google.com/file/d/1UUlKc9xfuapgLuVhCgPihbrx7Gd48avG/view)。private保存・再取得SHA・棋譜4局・optimizer・前後snapshot照合済み。次に200ms+長考1s×10の同条件で更新前後の直接比較、20runnerへのprivateデータ安全受渡しと長時間ハング対策を実装する。現時点で新たなActions runは開始していない。

## 2026-11-04：D40 100局完了・現行D37を維持

[D40：D37対D39・100局比較](D40_D37_VS_D39_100GAMES_20261104.md)を**50独立序盤×先後2局・計100局、14,535手合法再生**で完了。通常1手200ms、各陣営1秒長考最大10回、同じC++ NNUE探索。**D37＝50勝、D39＝49勝、未決着＝1（512手上限）**。D39の99決着局得点率49.49%で、50ペア単位の不確実性範囲は50%を含む。**D39の棋力向上は未確認のため、現行モデルはFUKAJI v0.4.0/D37(170d6c1e・Adam2106)を維持**。D39は学習後重み4c37e812・Adam2112として保全し、次周の親には使用しない。

実行はローカル4並列で、独立GitHub Actions 20ランナーは**未使用**。通常200ms指示の約13,935探索手中、D37側1手のみ330msの時間超過を記録。対局条件の絶対的時間上限保証は未達。[本人限定Drive100局完全記録](https://drive.google.com/file/d/1NxX0YOF0DHXjXW9MdmFJTMDuFahMYV3a/view?usp=drivesdk)をownerのみ・shared=falseで保存し、再取得SHA・全108ファイルハッシュを検証。D40検証完了で、前段[D39](D39_CURRENT_D37_SELFPLAY_REINFORCEMENT_20261104.md)の「学習後モデル比較待ち」は解消。採用後の新しい研究成果としては「強化未確認・現行D37維持」が結論。

## 2026-11-06：D41／20ランナー自己対局40局とD37の追加学習成功

[研究結果と実行証拠](D41_D37_20RUNNER_SELFPLAY_AND_TRAINING_20261106.md)。20独立GitHub Actionsジョブの[D41 run 42017791719](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42017791719)で**D37 vs D37 40局／5,351手**を全て合法検証し、公開[GitHub Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d41-d37-selfplay-42017791719)に保存。20対局ジョブ＋準備＋集約の計22ジョブ全成功。通常200ms／長考1s×各陣営最大10回、40局詰み決着。

初回の追加学習run 42018509500は`ModuleNotFoundError: fukaji`で失敗。インポート経路を修正して起動した[D41学習run 42120367483](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42120367483)は**成功**。D37からAdam**2106→2144**、新40局＋旧train40局で**2,398局面**、2epoch学習。学習後NNUE SHA256=`5cb31550fd9a011cace5b007df1c58c4e1889655e0a2f170869ce8eddb189e73`。[公開Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d41-d37-trained-42120367483)へ重み・optimizer・棋譜・結果を保存し、再ダウンロードSHA一致。

**残工程：D37更新前vs学習後モデルの通常200ms/1s長考条件での直接比較（独立50序盤×先後交替100局／20独立Actionsランナー）。** 比較による棋力向上はまだ未検証。**現行モデルはv0.4.0/D37（170d6c1e、Adam2106）のまま**で、強化が確認されるまで正式manifestは変更しない。本人の2026-11-05指示により新規成果の正本は**公開GitHub**とし、Driveとの二重保存をしない。既存Drive資産は移行検証完了まで残す。

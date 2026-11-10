# 深路：NNUE高速自己強化 — 現在地

2026-11-07更新。現行状態は以下。研究方針は[D17](D17_FAST_SELF_LEARNING_20261031.md)、判断正本は[decisions/](decisions/)、各比較結果はD系文書。ChatGPT・Claudeで同一Rule、Addness保留。

## 現行モデル

正式E8（18e6860405caf40c610c21c1b65851505372d6cd1734f720d3f71a271c746d30、Adam26573、E7から蒸留で更新）。E8 222-174(別シード400局・200ms)。[E8採用判断](decisions/2026-11-11_E8_E7親の蒸留NNUEを正式採用.md)。USI v0.4.1・PVS＋詰み探索不変。

## 現在の研究
- [D31](D31_TWO_BATCH_RETENTION_20261103.md)：400局/65,551手完了。c8対当時の正式NNUEは119勝76敗1分4未決着、対新経験のみ候補は97勝97敗4分2未決着。保持方式の候補間優位は未確定。
- [D33](D33_CANDIDATE_EXPERIENCE_20261103.md)：TRAIN64局（46決着）、Adam1580→1708の3seed追加学習。固定100序盤×先後**200局完了**。e0対親c8：113勝83敗1分3未決着、34,726手合法再生。95%区間43.17〜71.83%、50%を含み棋力差未確定。旧「起動拒否・待機」は履歴。
- [D35](D35_SELFPLAY_NNUE_CONTINUATION_20261103.md)：正式e0の学習状態から別領域で第16・17周自己対局・追加学習。研究latest db94693a/Adam1722。固定82序盤×先後164局：84勝72敗5分3未決着、31,018手合法再生。95%区間が50%を含み棋力差未確定。
- [D36](D36_SELFPLAY_NNUE_64_GAMES_20261103.md)は新32序盤・自己対局64局（56詰み、8未決着）を2回の追加学習につないだ。研究latest `7d74b86b`／Adam 1850。新20序盤を使った50ms比較で、学習前 `db94693a` に24勝16敗、正式 `e0cbd2ab` に22勝17敗1分。次の判断は[研究記録](decisions/2026-11-03_D36_研究用NNUE継続.md)。
- **過去（2026-11-04）の学習再開起点**は[D37完全state](https://drive.google.com/file/d/1XYvT7jA_0fqhPuo5v_uUIrxa4Crbocf-/view)（モデル170d6c1e / Adam2106）。**2026-11-07以降は冒頭のD43／Adam2882を現行再開点とする。**D36完全状態と停止indexは旧記録として保全。本人限定権限、再取得SHA、別領域実復元、両学習結果と比較結果の一致を確認。以前のD35再開点は[D35](D35_SELFPLAY_NNUE_CONTINUATION_20261103.md)に保全。

## 実行・判断の境界
最新main commit、稼働Actions、他AI/元writer、正式manifest、凍結plan、復旧archiveのSHAを確認。D17の結果重視学習とデータ分離を維持。NNUEは現行正式版から直接学習し、学習前snapshotとの直接対局を通過した重みを新正式版として次周へ継続する（2026-11-04決定、F00/F10/F20参照）。旧来の別研究候補と手動昇格の流れを次回学習に使わない。現行workflowが未対応なら先に実装を合わせる。少数局の故障確認は棋力実証と呼ばない。

## 過去の結果へのリンク
[D17](D17_FAST_SELF_LEARNING_20261031.md) / [D18](D18_RECOVERY_AND_LEARNING_20261101.md) / [D20](D20_RELIABLE_RESUME_AND_TARGET_TRIAL_20261101.md) / [D21](D21_KING_RELATIVE_NNUE_20261101.md) / [D22](D22_NNUE_QUIESCENCE_20261101.md) / [D23](D23_NATIVE_NNUE_SEARCH_20261101.md) / [D24](D24_FROZEN_NNUE_EVALUATION_20261101.md) / [D25](D25_REPLAY_RATIO_20261101.md) / [D26](D26_EXPERIENCE_DIVERSITY_20261101.md) / [D27](D27_PORTABLE_RECOVERY_20261102.md) / [D28](D28_NATIVE_ACCUMULATOR_20261102.md) / [D29](D29_STRATIFIED_REPLAY_20261102.md) / [D30](D30_FRESH_EXPERIENCE_20261102.md) / [D31](D31_TWO_BATCH_RETENTION_20261103.md) / [D32](D32_FORMAL_NNUE_RELEASE_20261102.md) / [D33](D33_CANDIDATE_EXPERIENCE_20261103.md) / [D34](D34_FORMAL_NNUE_RELEASE_20261103.md) / [D35](D35_SELFPLAY_NNUE_CONTINUATION_20261103.md)。[旧ROADMAP](archive/2026-10-31_ROADMAP_before_D17.md)は履歴。古い「現在」「次」は現行指示ではない。

[F01](rules/F01_継続実行とタイマー.md) / [F30](rules/F30_実行保存と再開.md) / [Drive容量候補台帳](operations/2026-11-03_Drive容量と保持候補.md)に従う。Addness Goalは保留中・履歴参照のみ任意。

## 2026-11-04追記：D37保存成果と次の正式比較

本人限定Driveの[D37最終アーカイブ](https://drive.google.com/file/d/1XYvT7jA_0fqhPuo5v_uUIrxa4Crbocf-/view)を実取得・読込し、次の実記録を確認した。
- **以下はD37採用前の2026-11-04時点の歴史的記録。当時はv0.4.0/D37が現行だったが、2026-11-07にD43へ更新済み**。当時の現行版はv0.3.0 / D33 NNUE e0cbd2ab（Adam1708）。D36は `7d74b86b`（Adam1850）、D37最終は `170d6c1e`（Adam2106）。D37はD36から各200局の自己対局・学習を2回行い、完全stateを保存。
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

## 2026-11-06 D42：学習前D37 vs 学習後D41を20ランナー100局比較、現行D37を維持

**実績：[D42全100局比較結果](D42_D37_VS_D41_100GAMES_RESULT_20261106.md)／[採否判断](decisions/2026-11-06_D42_学習前後100局比較と現行D37維持.md)**。

- [GitHub Actions #42218550652](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42218550652)は準備1＋**独立対局20**＋集約1の22ジョブ全件成功。**最大実同時稼働20**を各jobの時刻から照合。
- D37（学習前\`170d6c1e\`/Adam2106）対 D41（学習後\`5cb31550\`/Adam2144）を**新規独立50序盤×先後交替100局**、通常1手200ms・1秒長考は各陣営10回まで。**D37＝52勝、D41＝48勝、引分0・未決着0**。13,970手を合法再生。
- 学習後モデルは48%得点率で、事前に定めた差8勝以上の採用目安を満たさず、棋力向上未確認。**現行モデルはD37のまま維持**（\`formal/model.json\`は未変更）。D41学習後モデルは履歴として保存し、次周の学習親にしない。
- [完全棋譜・結果のGitHub Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d42-d37-before-vs-d41-after-42218550652)を**SHA \`5b2cb8495ae7fcdbe2b9181c3d5b3aedb87b2fd72226b1e4de76abb2d3d1f570\`**で再取得検証。
- 次工程は新しいD37自己対局・学習改善サイクル。重みの採用は学習前後の棋力比較で判断。GitHubを成果物正本とし新規Driveへの二重保存なし。

## 2026-11-06：D43／D37から1000局・20ランナー・学習・200局比較（当時の実行前計画）

[本人依頼とD43実験計画](D43_D37_1000_SELFPLAY_LEARNING_COMPARE_PLAN_20261106.md)を保存。現行D37/Adam2106を起点に、**新規1000局＝20個の独立Actionsランナー×各10局×5wave**、D37から直接継続学習、学習後に新規100序盤×先後200局で直接比較する。学習前のD37と過去比較群を保持し、全棋譜・重み・optimizerを公開GitHub Releaseに保存する構成。

**未実行**：新実行コードのGitHub作成が接続の安全性チェックにより拒否された。今回の1000局実job=0、自己対局=0、追加学習=0、比較=0。拒否を別API等で回避しない。D37現行モデル不変。権限が適切な環境でコード実装・実ジョブ起動・学習・検証・採否を進める。F01に従い実行中のD43 runがないためタイマー予約は設けない。

## 2026-11-07：D43実行の一次監査（当時の中間記録、D44で解消）

[D43 run 42292240321](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42292240321) の成果物一覧で5wave・学習・監査のアップロードを確認。jobs API先頭ページではprepare1件、wave0 20件、wave1 9件が成功。接続の100件ページ制限により、全100対局job、1000棋譜の独立性・合法性、学習後optimizer、200局勝敗、Release SHAは**未検証**。現行は引き続きD37/Adam2106、重み変更なし。[一次監査・次の一手](decisions/2026-11-07_D43_run42292240321_一次監査.md)。

## 2026-11-07：D44、D43学習後NNUEの現行化

- 本人指示によりD43(`bf8c26f9`/Adam2882)を現行モデルとして反映。[決定と採否根拠](decisions/2026-11-07_D43学習後モデルを現行に採用.md)。
- 推論用`models/fukaji-d43.nnue.npz`のSHA256：`f6ba1d5af46ae648b1921d675ada28cf5065f23a20e424154458817534bf42a6`。D43完全学習stateとoptimizerはGitHub Releaseに保存済み、旧D37は復元用に残す。
- 現行のUSI/version・探索コード・既定の探索時間は従来どおり。重みとoptimizerの進行状態だけを更新。正式契約テスト、C++ NNUE/USI動作と配布・SHAの完了証拠はD44 GitHub Actions runのログ／現行Releaseを参照。

## 2026-11-07：D44現行化の動作検証完了

[D44 GitHub Actions #42309719933](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42309719933)は成功。**713回帰テスト、正式契約、実C++ NNUEとUSIの動作、推論ファイルの再読込、別領域での配布パッケージ実起動、Release SHA再取得照合を完了。** 現行SHA `bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f`、Adam2882、[現行化の検証記録](D44_D43_CURRENT_ADOPTION_VERIFIED_20261107.md)、[公開モデル起動パッケージ](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-current-d43-20261107)、[採用コミット](https://github.com/mtk-ctrl/shogi-fukaji/commit/28f9d7ef0f3b77ac9086adae50e5b1f1162667e4)。将来の失敗時は更新前の**D43**に戻す。上記D37が現行だった記載は当時の記録。

## 2026-11-07：独創性のコンセプトと詰み探索を最優先

本人の最新判断：「**定跡は覚えさせない。AlphaGo Zero / AlphaZeroのように、自己対局と先読みで人間の常識にない手を発見する。ただし詰みは正確に読めるようにする。**」を深路の不変指針として採用。[決定の全文](decisions/2026-11-07_深路の独創性と詰み探索の原則.md)、規範[F00](rules/F00_深路の目的と権限.md)、学習境界[F10](rules/F10_NNUE学習とデータ境界.md)、検証[F20](rules/F20_研究更新と棋力検証.md)。現行D43/Adam2882と稼働中[D45第2周](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42326260509)は一切変えない。将来の改善は「合法手を全探索する詰み検出」と「自己対局からの自発的な方策・価値の向上」を優先し、人間の定跡や駒得などを正解教師として注入しない。勝率と詰みの見逃しを別々に実測する。研究ブランチ[D46](https://github.com/mtk-ctrl/shogi-fukaji/tree/research/d46-tactical-quality-20261107)の旧提案は本方針の制約下で再評価する。

## 2026-11-07：D45第2周は完了、学習後NNUEは採用基準通過（正式反映前）

- [D45 run 42326260509](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42326260509)は**success**。5波×20独立ジョブ＝新1000局、994局確定・6局未決着。単一writerでD43 `bf8c26f9`／Adam2882から学習後`941598f2`／**Adam3794**へ、58,334局面学習。
- 新100序盤×先後交替200局：**学習後155勝、学習前44勝、引分0、未決着1、差+111**。事前閾値差12・未決着10以下を通過。全125ジョブ成功、棋譜・学習state・比較の3 Release保存／CI内SHA読戻し完了。
- [D45詳細・3件のSHA](D45_D43_SECOND_ROUND_1000_GAMES_RESULT_20261107.md)、[採用ゲート判断](decisions/2026-11-07_D45第2周200局比較と採用ゲート.md)。
- **現行NNUEは依然D43 `bf8c26f9`／Adam2882**。D45比較は採用条件の棋力部分だけ通過。次はD44相当の契約・713件回帰・実USI・配布Release検査とmainの原子的反映。未検証のまま現行化しない。旧D43を保全する。USIと探索仕様は変えない。

## 2026-11-07：D46によるD45現行化

- 本人指定D45 `941598f2a9ec926d5f67dadf03582ccef718a56a843070dc1f576cc7eeca44e8`/Adam3794。200局155勝44敗・未決着1。強化後推論 `models/fukaji-d45.nnue.npz` SHA256 `9faf66a9d1fee9d282e35835628377ab7951bb94335a5d0498384bf88f4d2c26`。
- 変更はNNUE推論重みと現行モデル参照のみ。FUKAJI v0.4.0・探索不変。GitHub Actions D46の全契約・実USI・配布/読戻し完了とmain反映が採用完了条件。
- 完全重み・optimizerはD45学習Release、旧D43推論と完全stateは保持。次周唯一の親はD45/Adam3794。


## 2026-11-07：D48 詰み探索＋PVSの現行D45との100局（研究・正式採用なし）

- [D48実験計画・実測・次工程](D48_MATE_PVS_PAIRED100_20261107.md)、[採否判断](decisions/2026-11-07_D48詰み探索PVS100局結果と現行維持.md)、[実run](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42339652079)、[全原棋譜とSHA済みRelease](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d48-pvs-mate-d45-100games-42339652079)。
- 現行D45 NNUE同一重みで **PVS＋強制1/3/5手詰み＋相手1手詰みガードが55勝、現行αβが44勝、未決着1**。50独立序盤ペア100局、12,395手合法検証、20独立ランナー最大同時20、詰み証明113回。100局だけで優位確定とはしない。
- 現行NNUEは依然**D45/Adam3794**、正式FUKAJI v0.4.0の探索・USIは不変。研究用コード・Releaseのみ追加。次は詰み専用の未見問題で正答率・誤詰み・タイムアウトを独立監査し、実戦勝率を新規局面で再現確認する。

## 2026-11-07 D49：D48の探索を正式採用

本人明示指示で100局実測55勝44敗・未決着1のD48 PVS＋詰み補助を正式USIへ採用。現行USI FUKAJI v0.4.1、NNUE D45 snapshot 941598f2/Adam3794を不変維持。実装・契約検査・隔離配布・SHA検証を実施して正式化する。docs/decisions/2026-11-07_D49_D48探索を正式採用.md。


## D49正式探索採用・配布検証完了（2026-11-07）

- 正式source commit **408dc2994d6fc3d2cc69d9ce9a257cd157ae75c8**、[実GitHub Actions run 42341751230](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42341751230) **success**（713件回帰、5スキップ、実USI・詰み証明、隔離配布159ファイルSHA、Release4資産再取得SHA）。
- 現行は**FUKAJI v0.4.1、D45 NNUE不変、探索PVS＋詰み1/3/5手・相手の1手詰みガード**。[現行配布](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-current-d45-d49-pvs-mate-20261107)。採用理由と初回失敗の再発防止は[判断記録](decisions/2026-11-07_D49_D48探索を正式採用.md)。

## D50：D45/Adam3794・FUKAJI v0.4.1による1,000局強化（2026-11-07起動準備）

- 本人指示の第3周。現行NNUE D45（snapshot `941598f2...`・Adam3794）を唯一の親として、正式PVS＋詰み1/3/5手探索による1000局生成、継続学習、同じ探索での学習前後200局比較を一つの有限GitHub Actionsで実施。
- 5波×20独立ジョブ×各10局と比較20ジョブ。旧train/audit・D47/D48評価の局面を分離し、GitHub Releaseの原棋譜/完全学習stateをSHA再取得検証。性能を確認するまで正式重みは変更しない。強化が認められた後は別検証ワークフローでNNUEのみ採用し、探索／USI v0.4.1は維持。
- [固定計画](D50_CURRENT_D45_V041_1000GAMES_PLAN_20261107.md)／[本人判断](decisions/2026-11-07_D50第3周1000局の実行判断.md)／[workflow](../.github/workflows/d50-d45-v041-1000-selfplay-train-audit.yml)。

## D51：D50の正式採用（2026-11-07）

- [D50学習・200局直接比較](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42343709475)：1000局全決着、58,077学習局面、Adam3794→4702、**D50 159勝D45 41勝、未決着0**。事前基準通過。
- 正式NNUEを `641f36677b8d93c3237cfc4f0b6329a6beee55cff692302f529b3899ea74afd0` へ更新（推論SHA `fbabd6b98de1d03ff8755733bb6a5a5daac7a0fd1f2f2c11ff2b21c26836c599`）。D49のPVS＋詰み探索とUSI v0.4.1は維持。[採用判断](decisions/2026-11-07_D51_D50学習後重みを現行採用.md)。契約・回帰・実USI・隔離bundle・Release SHAの通過をD51 Actionsで必須とし、正式配布の読戻しを検証する。

## D51正式採用・配布検証完了（2026-11-07）

- [D51 Actions run 42349125449](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42349125449) **success**（07:00:48 UTC、全16ステップ成功）。正式採用source commit `cd31eb382bb6bb283991cc88d9da660f20dc8b45`。main `formal/model.json` の読戻しで**現行NNUE D50、Adam4702**を確認。
- 正式推論 `models/fukaji-d50.nnue.npz` SHA256 `fbabd6b98de1d03ff8755733bb6a5a5daac7a0fd1f2f2c11ff2b21c26836c599`。正式C++ PVS＋1/3/5手詰み探索SHA `c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a`、USI v0.4.1は**不変**。旧D45重み・optimizerも復元用に保持。
- [正式配布Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-current-d50-d51-pvs-mate-20261107)：bundle SHA256 `6af46533ae99cef0011b277899aacade90f77f18571735e2f1c1ec6b234a1495`。実USI合法60手、1手詰み、隔離159ファイル整合、4資産再取得SHA一致を検証。[D50詳細結果](D50_CURRENT_D45_V041_1000GAMES_PLAN_20261107.md)、[採用判断](decisions/2026-11-07_D51_D50学習後重みを現行採用.md)。次周はD50完全state/Adam4702からのみ継続。

## D52：D51正式採用後の現行D50/Adam4702から第4周1000局（2026-11-07）

- 本人の条件付き指示「正式採用できたら、また1000局」に対し、D51正式採用成功を検証してD52を実行。D50 NNUE（641f3667…）/Adam4702 と FUKAJI v0.4.1、PVS＋詰み1/3/5手探索は維持。
- 5波×20独立ランナー×10局＝1,000新局、学習後の200局標準比較を通じてさらに棋力を上げる。**過去D50の訓練1,000局と評価200局を明示除外**し、学習は正式D50完全optimizerから直接再開。
- [D52計画](D52_CURRENT_D50_V041_1000GAMES_PLAN_20261107.md)／[本人判断](decisions/2026-11-07_D52第4周1000局開始判断.md)／[起動workflow](../.github/workflows/d52-d50-v041-1000-selfplay-train-audit.yml)。正式重みの更新は200局勝率と契約・実USI・SHAを検証した後の別工程に限定。

- **2026-11-07 17:07 JST D52進捗：** [run 42350699929](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42350699929) で第1～4波（wave0～3）80/80ジョブ成功、第5波（wave4）2/20成功・17実行中・1待機、失敗0。**820局分の生成ジョブ成功、全局合法性最終検証は未確認**。1000局集約→NNUE学習→200局比較→条件を満たす場合のみ別CIで正式採用。正式D50は不変。詳細は[D52計画](D52_CURRENT_D50_V041_1000GAMES_PLAN_20261107.md)。

## D53：D52 NNUE正式採用（2026-11-08）

- [実行run](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42401026786)：1000局全局決着、57,755局面・Adam4702→5606、200局比較で新D52が156勝44敗、未決着0、19,104手合法検証。学習・比較の全3 ReleaseのSHA再取得成功。
- 正式NNUE 55ccc97db6c2d4ce9ff7c75f5e81d543fb6f12d169639c3ea6d24c89e95342ae、推論SHA c0bfe9640e8fc8d6750d263f367581d33c428c7c05cea99f384afaeabfbb569f。PVS＋詰み探索・USI v0.4.1不変。契約・回帰・実USI・隔離配布を別D53 CIで検証。[判断](decisions/2026-11-08_D53_D52学習後重みを現行採用.md)。

## D53正式採用・配布検証完了（2026-11-08）

- [D53 run 42424484457](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42424484457) **success**。現行NNUEはD52 snapshot `55ccc97db...`／Adam5606、推論重み `models/fukaji-d52.nnue.npz` SHA256 `c0bfe9640e8fc8d6750d263f367581d33c428c7c05cea99f384afaeabfbb569f`。正式USI v0.4.1とPVS＋1/3/5手詰み探索SHA `c9340af2...` は不変。
- [正式配布4資産](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-current-d52-d53-pvs-mate-20261108) は公開後に4件SHA再取得照合成功。bundle SHA256 `beb57882ea49744e46b45142303a0741cb6220138a912d9c461af9efd099cc1d`。隔離bundleで実USI 2局60合法手・32局面depth2同一・詰み証明通過。旧D50保全。
- [検証判断と全資産SHA](decisions/2026-11-08_D53_D52学習後重みを現行採用.md)。次の強化は採用済みD52/Adam5606の完全学習stateから進める。D53採用の再実行は不要。

## D54：現行D52/Adam5606から第5周 自己対局1,000局（2026-11-08）

- 本人明示指示。D53正式採用のD52 NNUE（snapshot 55ccc97d...、Adam5606）を唯一の学習親とし、5波×20ランナー×10局＝新規1,000局、旧trainの許可済みreplay、未見100序盤の先後200局を同じv0.4.1 PVS+詰み探索で直接比較。事前基準勝数差12以上/未決着10以下。
- [固定実験計画](D54_CURRENT_D52_V041_1000GAMES_PLAN_20261108.md)、[判断と境界](decisions/2026-11-08_D54第5周1000局の実行判断.md)、[workflow](../.github/workflows/d54-d52-v041-1000-selfplay-train-audit.yml)。前回の101artifact欠損、Release403をコードで防止。研究run完了/対局結果は開始後の実ログでのみ判定し、正式D52は検証後の別採用手続まで不変。

## D55：D54 NNUE正式採用（2026-11-08）

- 1000局全局決着、57615局面でAdam5606→6508、別の200局比較では新D54 137勝、D52 62勝、未決着1。合計20,297手合法検証。3件のReleaseのSHA再取得確認。
- 正式NNUE 1cb833c18be57d4881388bb6947f17493cca4974faac4023a4550ea47463f3f5、推論SHA 5c2e011225e914d95b36afaecf56fd267db273a325de47bf371eaa93e8dc36a2。USI v0.4.1・PVS＋詰み探索維持。[判断](decisions/2026-11-08_D55_D54学習後重みを現行採用.md)。

## D57正式採用（2026-11-08）

- D56 1000局中997決着、Adam6508→7402、独立200局135勝65敗・未決着0。正式NNUE 89562fd4fed556f4a883dba465add35c532cc2c7ccdae9efd24d685270246363、推論SHA bcc1da5a3756c9a70fd864add32571a5b289a47d3614f794eed4fcf9802d5d24。v0.4.1とPVS＋詰み探索不変。
- [判断記録](decisions/2026-11-08_D57_D56学習後重みを現行採用.md)。

## D59：D58正式採用（2026-11-08）

D58は新1000局999決着、Adam7402→8296、直接200局121勝79敗・未決着0・合法18,739手。正式推論SHA d856765813508a29cb2f35e2aceba8b67dc7797798eb114ae23d826424baf30d。[採用](decisions/2026-11-08_D59_D58学習後重みを現行採用.md)。

## D61：D60正式採用（2026-11-08）

D60は新1000局997決着、Adam8296→9190、直接200局123勝77敗・未決着0・合法18,815手。正式推論SHA 6cabbceb047ce19b527ae7261235ee84fca43fb42fd3af745872cbb330fd3257。[採用](decisions/2026-11-08_D61_D60学習後重みを現行採用.md)。

## D63：D62正式採用（2026-11-08）

D62は新1000局996決着、Adam9190→10080、独立200局125勝74敗・引分1・未決着0・合法18,033手。正式推論SHA 3df737429572f27470e9f7a495cb772e569ed60135e3a1b6badfef53f67636ce。[採用](decisions/2026-11-08_D63_D62学習後重みを現行採用.md)。

## D65：D64正式採用（2026-11-08）

D64は新1000局999決着、Adam10080→10974、独立200局114勝86敗・合法17,324手。推論SHA fd5f832e0ded9519c23ad7469eee7a79f70d78f41bc0dab87c5f7bc57c8d478a。[採用](decisions/2026-11-08_D65_D64学習後重みを現行採用.md)。

## D67：D66正式採用（2026-11-08）

正式D66（c29b5ba1ed9264ba298ffdef2ab0deb71f37b327a32c3a1f8505061f67401ac3、Adam11864）。D66 118勝・D64 81勝・引分0・未決着1。[D67判断](decisions/2026-11-08_D67_D66学習後NNUEを正式採用.md)。次周唯一親は[D66完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d66-d64-trained-42471868758)。USI v0.4.1・PVS＋詰み探索不変。

## E3：雲路練習局面学習NNUE正式採用（2026-11-09）

正式E3（193bc0e2ce961284958660f8709473551b3e31a42e352a5b88a76ebce41d27ac、Adam12562、D66から雲路練習局面で更新）。E3 114-86(200局)・E3 212-186(確認400局)。[E3採用判断](decisions/2026-11-09_E3_雲路練習局面学習NNUEを正式採用.md)。USI v0.4.1・PVS＋詰み探索不変。

## E2：探索評価値蒸留NNUE正式採用（2026-11-09）

正式E2（fd0bae3867b0af0d4b404af857e6856ba97b29cc2adb323b3977b33d7872c6f5、Adam13394、E3から探索評価値の蒸留で更新）。E2 121-77(200局)・E2 240-158(確認400局)。[E2採用判断](decisions/2026-11-09_E2_探索評価値蒸留NNUEを正式採用.md)。USI v0.4.1・PVS＋詰み探索不変。

## E4：規模拡大蒸留NNUE正式採用（2026-11-10）

正式E4（aa2399ba25920e24ac49bcfcf10093a867ce4861b7afda4de3bd4f7a6db4e267、Adam15680、E2から規模拡大蒸留で更新）。E4 121-78(200局)・E4 231-167(確認400局)。[E4採用判断](decisions/2026-11-10_E4_規模拡大蒸留NNUEを正式採用.md)。USI v0.4.1・PVS＋詰み探索不変。

## E5：E4親の蒸留NNUE正式採用（2026-11-10）

正式E5（638795c217155c86392affbd8857d07bfc6d58f1e36f1fd4fd5ccf69a5556d23、Adam18074、E4から蒸留で更新）。E5 120-79(200局)・E5 248-150(確認400局)。[E5採用判断](decisions/2026-11-10_E5_E4親の蒸留NNUEを正式採用.md)。USI v0.4.1・PVS＋詰み探索不変。

## E6：隠れ層128NNUE正式採用（2026-11-10）

正式E6（ad6eed6de3c2e86c1ed68c62dfe2d86ea1eba6f7ebd5cc20ffbe5de5ed268e58、Adam21671、隠れ層128。E5から幅を広げて蒸留）。E6(隠れ層128) 122-78(対E5 200局)。[E6採用判断](decisions/2026-11-10_E6_隠れ層128NNUEを正式採用.md)。USI v0.4.1・PVS＋詰み探索不変。

## E7：E6親の蒸留NNUE正式採用（2026-11-10）

正式E7（d139753b3c23317255b8e8be9f97e6ed42d8c35312998dc12ebf5443b52385dc、Adam24127、E6から蒸留で更新）。E7 117-83(200局)。[E7採用判断](decisions/2026-11-10_E7_E6親の蒸留NNUEを正式採用.md)。USI v0.4.1・PVS＋詰み探索不変。

## E8：E7親の蒸留NNUE正式採用（2026-11-11）

正式E8（18e6860405caf40c610c21c1b65851505372d6cd1734f720d3f71a271c746d30、Adam26573、E7から蒸留で更新）。E8 222-174(別シード400局・200ms)。[E8採用判断](decisions/2026-11-11_E8_E7親の蒸留NNUEを正式採用.md)。USI v0.4.1・PVS＋詰み探索不変。

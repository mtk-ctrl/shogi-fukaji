# 深路：ChatGPT・Claude共通の作業入口

**2026-11-17本人決定（最優先で読む）**：①対現行最新版で勝敗差+12以上（標準108局）かつ違法手0なら、承認を待たず正式採用まで自動で実行し事後報告（NNUE重みに限らずエンジン変更も同じ）。②本人確認で止めるのは削除を伴う操作だけ。次の工程はAIが選んで起動し、確認予約を作ってから終える。③Floodgateは雲路・深路とも常時参加を継続し、採用した新版は次の回から自動で使われる。④ランナーは両repo合計18台まで（2台はFloodgate用に常に空ける）、空き時間を作らない・1job60分以内。⑤人生帳（mtk-ctrl/lifeplan）の読み書き・公開は常に許可。詳細は[判断記録](docs/decisions/2026-11-17_自動採用と自走運用.md)。

ChatGPT・Claude等は同じRuleと同じGitHub正本を使う。まず[開発ルーター](docs/00_開発ルーター.md)、[F00](docs/rules/F00_深路の目的と権限.md)、[ROADMAP](docs/ROADMAP.md)を読む。共通原則はGoogle Driveの `00_AI運用基盤/00_AI共通ルーター.md`（名前が「日付_旧_」で始まらない現行版）の第5節（判断記録の正本）・第6節（複数AI）。

深路 `mtk-ctrl/shogi-fukaji` と雲路 `mtk-ctrl/shogi-ai` は独立。雲路の手作業評価配点・重み・研究手・局面知識を教師として移植しない。合法手・探索・時間管理の一般技術は深路で独立検証。

判断・理由・失敗・検証・次の一手は `docs/decisions/`、研究の計画と結果はD系文書、現在地は `docs/ROADMAP.md`、正式実装は `formal/model.json` と現行仕様を正本とする。**Addnessは2026-11-03から利用保留**。既存Goalは過去経緯の参照用のみ任意、書込み・読戻しを完了条件にしない。

研究方針[D17](docs/D17_FAST_SELF_LEARNING_20261031.md)と[F00](docs/rules/F00_深路の目的と権限.md)を継承する。NNUE学習はその時点の正式版重みを直接の起点とし、学習前の重みを不変の比較・復元用snapshotとして保全する。学習前後を直接対局し、比較計画の条件で強化を確認できた重みは新しい正式版として登録して次周の起点にする。独立した研究候補モデル系列を作らない。**現行NNUE重みはD45（941598f2、Adam3794）。USI表示はFUKAJI v0.4.0を維持**。2026-11-04に本人がD37を直接現行モデルとして採用。[決定](docs/decisions/2026-11-04_D37を現行モデルに採用.md)と[現行仕様](docs/現行エンジン仕様.md)を参照。正式USIや探索仕様の変更は別途明示指示なしに行わない。

## NNUE正式版を起点にする学習手順（2026-11-04決定）

- 学習開始時点の正式版NNUE重みを、そのまま次の学習の親として使う。別系統の研究候補モデルを起点にしない。
- 学習前の正式版重みは、比較対局と失敗時の復元に使う不変snapshotとして保存する。これは独立した研究モデルではなく、正式版の学習前状態である。
- 学習で更新した重みを同じ実行サイクル内で学習前snapshotと直接比較する。探索設定、実時間、CPU、序盤、先後をそろえ、対局結果・未決着・合法性を記録する。
- 学習後の重みが比較計画の判定条件で強くなったら、その重みを新しい正式版として登録し、以後の学習の起点にする。改善が確認できなければ学習前の正式版へ戻す。結果が判然としない場合は正式版を維持して比較を追加する。
- 比較・採用は学習前後の正式版重み間で行い、研究latestや別候補との間接比較で置き換え可否を判断しない。
- 正式版の学習前snapshot、学習後重み、optimizer、棋譜、比較条件・結果、各SHAを保存する。更新は原子的に行い、途中失敗で正式重みを壊さない。
- この運用はNNUE重みの更新に適用する。探索コード・USI既定値・配布形式など別の正式仕様変更は、別途明示指示がない限り変更しない。

長時間の自律実行・予約実行・通知は[F01](docs/rules/F01_継続実行とタイマー.md)、保存・再開は[F30](docs/rules/F30_実行保存と再開.md)。同等機能がないAIは有限のGitHub Actions、GitHub結果記録と本人限定Drive保存、次回対話での再開を使い、自動実行済みと偽らない。

作業前・push直前に最新main commit・Actions・他AIの研究writer・対象ファイルを確認。競合時は別ブランチ/PRで提案し、旧版で上書きしない。文書変更もworkflow起動条件と研究保存を確認。D27/D30のChatGPT Work/Automations障害の記録は事実として保全する。

## 最新の正式探索（2026-11-07 D49）

本人はD48のPVS＋詰み探索を正式採用と明示。現行は FUKAJI v0.4.1、D48と同じPVS＋1/3/5手詰み証明＋相手1手詰みガード。NNUEはD45/Adam3794を維持。上部のv0.4.0はD49以前の履歴。判断記録：docs/decisions/2026-11-07_D49_D48探索を正式採用.md。

## 最新現行NNUE（D51、2026-11-07）

**現行はD50/Adam4702、snapshot `641f36677b8d93c3237cfc4f0b6329a6beee55cff692302f529b3899ea74afd0`。** 現行NNUE D45（Adam3794）から自己対局1000局＋比較200局で159勝41敗となり、F00/F20の本人委任に従ってNNUE重みのみ正式反映。**FUKAJI v0.4.1、D49のPVS＋1/3/5手詰み探索はそのまま**。上部のD45/旧版表示はその時点の歴史的記述。次周は[D50完全学習state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d50-d45-trained-42343709475)のAdam4702を唯一の学習親とする。[判断](docs/decisions/2026-11-07_D51_D50学習後重みを現行採用.md)。

## 最新正式NNUE（D53、2026-11-08）

**正式D52/Adam5606、snapshot 55ccc97db6c2d4ce9ff7c75f5e81d543fb6f12d169639c3ea6d24c89e95342ae。** D50/Adam4702から自己対局1000局→学習、比較200局でD52 156勝44敗、引分0・未決着0。F00/F20に従いNNUE重みのみ更新。FUKAJI v0.4.1と正式PVS＋詰み探索は不変。次周唯一の親は[D52完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d52-d50-trained-42401026786)。旧D50系の記述はその時点の履歴。[D53判断](docs/decisions/2026-11-08_D53_D52学習後重みを現行採用.md)。

## 最新正式NNUE（D55、2026-11-08）

**正式D54/Adam6508、snapshot 1cb833c18be57d4881388bb6947f17493cca4974faac4023a4550ea47463f3f5。** 旧正式D52/Adam5606から1000局→学習、200局で137勝62敗、未決着1。NNUE重みのみ更新、PVS＋詰み探索・v0.4.1不変。次周唯一の親は[D54完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d54-d52-trained-42426372939)。旧モデル説明はその当時の履歴。[D55判断](docs/decisions/2026-11-08_D55_D54学習後重みを現行採用.md)。

## 最新正式NNUE（D57、2026-11-08）

正式D56/Adam7402、snapshot 89562fd4fed556f4a883dba465add35c532cc2c7ccdae9efd24d685270246363。旧D54から1000局学習、200局135勝65敗・未決着0。NNUE重みのみ更新、USI v0.4.1とPVS＋詰み探索不変。次周の親は[D56完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d56-d54-trained-42433340061)だけ。[判断](docs/decisions/2026-11-08_D57_D56学習後重みを現行採用.md)。

## 最新正式NNUE（D59、2026-11-08）

正式D58/Adam8296、snapshot d190c1a1b8d5a44b3fe4791588306fb493f1b9534cca268fcbb656ba8604265c。旧D56/Adam7402から1000局学習、先後200局121勝79敗、未決着0。NNUEのみ変更、USI v0.4.1とPVS＋詰み探索不変更。次周は[D58完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d58-d56-trained-42439684313)を唯一の親とする。[D59判断](docs/decisions/2026-11-08_D59_D58学習後重みを現行採用.md)。

## 最新正式NNUE（D61、2026-11-08）

正式D60/Adam9190、snapshot 1f4ef9f32dfff7a94ad3ddb1c57f64f123ca8b6f6caa422cb9c86ace06c57654。旧D58/Adam8296から1000局学習、先後200局123勝77敗、未決着0。NNUEのみ変更、USI v0.4.1とPVS＋詰み探索不変更。次周は[D60完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d60-d58-trained-42447988443)を唯一の親とする。[D61判断](docs/decisions/2026-11-08_D61_D60学習後重みを現行採用.md)。

## 最新正式NNUE（D63、2026-11-08）

正式D62/Adam10080、snapshot e04e85dfebb507131df591bf0206e27771afe6944f3249124dfaf6eab5de51fb。D60/Adam9190から新1000局学習、独立200局125勝74敗、引分1、未決着0。NNUEのみ変更、USI v0.4.1とPVS＋詰み探索は不変更。次周は[D62完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d62-d60-trained-42458207248)を唯一の親とする。[D63採用判断](docs/decisions/2026-11-08_D63_D62学習後重みを現行採用.md)。

## 最新正式NNUE（D65、2026-11-08）

正式D64/Adam10974、snapshot 545dba627046621ec53b681d485978b9054776c717d894dcb0de55bec731d4a2。旧D62/Adam10080から1000局学習、200局114勝86敗、未決着0。NNUEのみ変更、USI v0.4.1・PVS+詰み探索不変。次周は[D64完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d64-d62-trained-42465412656)を唯一親とする。[D65判断](docs/decisions/2026-11-08_D65_D64学習後重みを現行採用.md)。

## 最新正式NNUE（D67、2026-11-08）

正式D66／Adam11864、snapshot c29b5ba1ed9264ba298ffdef2ab0deb71f37b327a32c3a1f8505061f67401ac3。D66 118勝・D64 81勝・引分0・未決着1。学習前D64を保全、NNUEのみ採用。[完全state](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d66-d64-trained-42471868758)。

## 最新正式NNUE（E3、2026-11-09）

正式E3／Adam12562、snapshot 193bc0e2ce961284958660f8709473551b3e31a42e352a5b88a76ebce41d27ac。E3 114-86(200局)・E3 212-186(確認400局)。NNUEのみ更新、旧D66を保全。

## 最新正式NNUE（E2、2026-11-09）

正式E2／Adam13394、snapshot fd0bae3867b0af0d4b404af857e6856ba97b29cc2adb323b3977b33d7872c6f5。E2 121-77(200局)・E2 240-158(確認400局)。NNUEのみ更新、旧E3・旧D66を保全。

## 最新正式NNUE（E4、2026-11-10）

正式E4／Adam15680、snapshot aa2399ba25920e24ac49bcfcf10093a867ce4861b7afda4de3bd4f7a6db4e267。E4 121-78(200局)・E4 231-167(確認400局)。NNUEのみ更新、旧E2・旧E3・旧D66を保全。

## 最新正式NNUE（E5、2026-11-10）

正式E5／Adam18074、snapshot 638795c217155c86392affbd8857d07bfc6d58f1e36f1fd4fd5ccf69a5556d23。E5 120-79(200局)・E5 248-150(確認400局)。NNUEのみ更新、旧E4・旧E2・旧E3・旧D66を保全。

## 検証の進め方（2026-11-10）

検証時間は他の作業の損失になる。明らかな優位が出たら追試せず先へ進み、基準すれすれ・方向が割れるときだけ確かめる。詳細は docs/検証の進め方.md。

## 最新正式NNUE（E6、2026-11-10）

正式E6／隠れ層128／Adam21671、snapshot ad6eed6de3c2e86c1ed68c62dfe2d86ea1eba6f7ebd5cc20ffbe5de5ed268e58。E6(隠れ層128) 122-78(対E5 200局)。構造変更あり（kNHidden・契約更新）。旧E5以前はmodels/に保全（隠れ層32）。

## 最新正式NNUE（E7、2026-11-10）

正式E7／Adam24127、snapshot d139753b3c23317255b8e8be9f97e6ed42d8c35312998dc12ebf5443b52385dc。E7 117-83(200局)。NNUEのみ更新、旧E6・旧E2・旧E3・旧D66を保全。

## 最新正式NNUE（E8、2026-11-11）

正式E8／Adam26573、snapshot 18e6860405caf40c610c21c1b65851505372d6cd1734f720d3f71a271c746d30。E8 222-174(別シード400局・200ms)。NNUEのみ更新、旧E7・旧E2・旧E3・旧D66を保全。

## 最新正式NNUE（E9、2026-11-11）

正式E9／Adam27469、snapshot c9302b9552d90fdfe88ac38ca54970fa3eceeaf67c024368f13d12b87af2b050。E9 321-275(対E8 合算600局: 最初の200局111-88＋別シード400局210-187・200ms)。NNUEのみ更新、旧E8・旧E7・旧E2・旧E3・旧D66を保全。

## 最新正式NNUE（E10、2026-11-13）

正式E10／Adam29173、snapshot c7eb865f9800bc75c7067c37d69a750ae9b32a824d1ddeb21f5a3bc9ff60a72f。E10 ハイブリッド探索下(雲路の探索＋深路NNUE, 1手2秒相当・同一探索)で対E9 83-17(100局)、対雲路最新 16-84(100局。E9は4-96)。NNUEのみ更新、旧E9以前を保全。

## 最新正式NNUE（E11、2026-11-14）

正式E11／Adam49207、snapshot 9f1b65eb0b98af31400ad05bdff471690121d03ec7ef748e2ee067442c611cd3。E11 ハイブリッド(雲路の探索＋深路NNUE, 200ms・同一探索・各100局)で対E10 84-16、対雲路 36-64(E10は16-84)、同じデータで現行特徴のまま学習した候補に 64-36。正式エンジン(ハイブリッド)の重みは formal/model.json の hybrid節(`models/fukaji-e11.nnue.npz`)。旧エンジン(legacy)はE10のまま。次の学習は g1-gen.yml → g1-train.yml(親E11) → hybrid-match.yml。

## 最新正式NNUE（E12、2026-11-15）

正式E12／Adam70555、snapshot 797277c909a8f2e74521aab049021b5234f3eea21036a9a1b41d4e0b4ce1ebee。E12 ハイブリッド(200ms・同一探索・各100局)で対E11 64-35-1、対雲路 49-51。正式エンジン(ハイブリッド)の重みは formal/model.json の hybrid節(`models/fukaji-e12.nnue.npz`)。旧エンジン(legacy)はE10のまま。

## 最新正式NNUE（E13、2026-11-15）

正式E13／Adam91824、snapshot 6b91fba9a68cbb74eeacdd7595f2b9d3c9347e994b5a0794bcb6146d75dc0c0f。E13 ハイブリッド(200ms・同一探索・各100局)で対E12 63-36-1、対雲路 56-44。正式エンジン(ハイブリッド)の重みは formal/model.json の hybrid節(`models/fukaji-e13.nnue.npz`)。旧エンジン(legacy)はE10のまま。

## 評価値の目盛り2400・左右反転学習（2026-11-15）

正式エンジン(ハイブリッド)のcp換算は hybrid.nnue_scale=2400(600から変更。同じE12で2400対600＝66-42、対雲路53-55)。学習は左右反転を既定で使う(g1-train.yml mirror=true。反転あり候補は反転なし候補に64-43-1)。生成データは自分の目盛りを summary の nnue_scale に持ち、学習はそれに従う(記録なし＝600)。正式NNUEは別作業でE13(周回3の反転なし候補)に更新済み。反転あり候補(学習run 43664202193)はE12が親の兄弟候補で、未採用。[判断](docs/decisions/2026-11-15_評価値の目盛り2400と左右反転学習を採用.md)。

## 雲路の探索を最新版に更新・探索の既定値を調整（2026-11-15）

正式エンジン(ハイブリッド)の探索は雲路 199dbb0(2スレッド・置換表改善)、既定値は QDelta=400・NullMoveReduction=4・LmrMinDepth=2・LmrReduction=2(2026-11-16に更新。旧: QDelta=600・NullMoveReduction=3。formal/model.json の hybrid.search_defaults、build_hybrid.py が当てる)。同じE13・108局で、固定版(1e08de5)に76-32、最新の既定値に74-34。NNUEの重み・評価の差し替えは不変。[判断](docs/decisions/2026-11-15_やり方を変える_周回をやめて探索側の調整と学習データの足し合わせへ.md)。探索の設定を比較するときは hybrid-tune.yml(候補側だけオプションを足して同じNNUEで対局)。

## 最新正式NNUE（E14、2026-11-15）

正式E14／Adam326414、snapshot 04de47d77db5a472e93d2e5261387a97bce366bdc2e13932b668c56373711c47。E14 ハイブリッド(200ms・同一探索・各108局)で対E13 67-40-1、対雲路 77-31。正式エンジン(ハイブリッド)の重みは formal/model.json の hybrid節(`models/fukaji-e14.nnue.npz`)。旧エンジン(legacy)はE10のまま。

## 最新正式NNUE（E15、2026-11-17）

正式E15／Adam644114、snapshot 5401b38da6fa60c18cc4f587b9b86fb3458b78bf300040ca15912930a7734683。E15 ハイブリッド(200ms・同一探索・各108局)で対E14 61-45-2、対雲路 88-20。正式エンジン(ハイブリッド)の重みは formal/model.json の hybrid節(`models/fukaji-e15.nnue.npz`)。旧エンジン(legacy)はE10のまま。

## 最新正式NNUE（E16、2026-11-17）

正式E16／Adam1044744、snapshot a424d2a16694a9ad178187bc0fb27f6dfa24d52f551a70424b71d6a98569f910。E16 ハイブリッド(200ms・同一探索・各108局)で対E15 64-44、対雲路 93-15。正式エンジン(ハイブリッド)の重みは formal/model.json の hybrid節(`models/fukaji-e16.nnue.npz`)。旧エンジン(legacy)はE10のまま。

## 最新正式NNUE（E17、2026-11-17）

正式E17／Adam1243977、snapshot 6f78a581fec1be9d51c6414870230556ec01fb3d177eb7df5bc429047060aa6a。E17 ハイブリッド(200ms・同一探索・各108局)で対E16 76-32、対雲路は省略。正式エンジン(ハイブリッド)の重みは formal/model.json の hybrid節(`models/fukaji-e17.nnue.npz`)。

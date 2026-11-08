# 深路：ChatGPT・Claude共通の作業入口

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

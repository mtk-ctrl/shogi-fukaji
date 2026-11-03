# 深路：ChatGPT・Claude共通の作業入口

ChatGPT・Claude等は同じRuleと同じGitHub正本を使う。まず[開発ルーター](docs/00_開発ルーター.md)、[F00](docs/rules/F00_深路の目的と権限.md)、[ROADMAP](docs/ROADMAP.md)を読む。共通原則は[Drive 00_AI共通ルーター](https://drive.google.com/file/d/1Kft2Uyld-m4b48Dp4e3P4BzYfBH96waY/view)第5節（判断記録の正本）・第6節（複数AI）。

深路 `mtk-ctrl/shogi-fukaji` と雲路 `mtk-ctrl/shogi-ai` は独立。雲路の手作業評価配点・重み・研究手・局面知識を教師として移植しない。合法手・探索・時間管理の一般技術は深路で独立検証。

判断・理由・失敗・検証・次の一手は `docs/decisions/`、研究の計画と結果はD系文書、現在地は `docs/ROADMAP.md`、正式実装は `formal/model.json` と現行仕様を正本とする。**Addnessは2026-11-03から利用保留**。既存Goalは過去経緯の参照用のみ任意、書込み・読戻しを完了条件にしない。

研究方針[D17](docs/D17_FAST_SELF_LEARNING_20261031.md)を継承する。研究コードのmain反映、研究用暫定モデルの更新、本人判断による正式採用を区別。現在の正式版はv0.3.0/D33 NNUE。正式USI・モデル・固定基準をAIの判断だけで変更しない。

長時間の自律実行・予約実行・通知は[F01](docs/rules/F01_継続実行とタイマー.md)、保存・再開は[F30](docs/rules/F30_実行保存と再開.md)。同等機能がないAIは有限のGitHub Actions、GitHub結果記録と本人限定Drive保存、次回対話での再開を使い、自動実行済みと偽らない。

作業前・push直前に最新main commit・Actions・他AIの研究writer・対象ファイルを確認。競合時は別ブランチ/PRで提案し、旧版で上書きしない。文書変更もworkflow起動条件と研究保存を確認。D27/D30のChatGPT Work/Automations障害の記録は事実として保全する。

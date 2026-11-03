# 2026-11-03 深路のChatGPT・Claude共通RuleとAddness保留

本人判断：ChatGPTとClaudeは同じRule。Addness利用保留。Drive[00_AI共通ルーター](https://drive.google.com/file/d/1Kft2Uyld-m4b48Dp4e3P4BzYfBH96waY/view)第5節・第6節に従う。

- 確定判断・理由・検証・失敗原因・次の一手はGitHub `docs/decisions/`、D系、ROADMAPへ。Addness既存Goalは保留中・過去参照のみ任意。
- 長時間自律実行、予約、通知は機能で定義。同等機能なしなら有限Actions→GitHub/本人限定Driveへ結果・完全state・停止済みindex保存→次回対話で復元。
- 正式版はv0.3.0/D33代表。D33 followup200は113勝83敗1分3未決着で完了。D35は第16・17周と164局比較完了（84勝72敗5分3未決着）。強さの優位性は未確定。
- 旧D19のDriveフォルダをID固定のままD19〜D35の実態に合わせ改名。F30で直近2中間checkpoint、各研究最終、正式採用版、重要開始前stateの保持条件を定め、[整理候補26件の容量台帳](../operations/2026-11-03_Drive容量と保持候補.md)を作成。削除なし。
- 本修正は文書・フォルダ名称のみ。学習器、対局、モデル、workflow、研究stateは未変更。D27/D30のChatGPT Work/Automations障害は事実として保全する。

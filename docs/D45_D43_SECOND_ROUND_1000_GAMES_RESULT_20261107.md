# D45 第2周：実行結果と現行化判定（2026-11-07）

## 確認対象と状態

- GitHub Actions [run 42326260509](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/42326260509) は **2026-11-07 03:07 UTC（日本時間12:07）にsuccess完了**。APIの jobs 1〜2ページを確認：全125ジョブがsuccess。内訳は prepare 1、自己対局wave0〜4各20＝100、aggregate 1、train 1、audit_prepare 1、監査audit 20、finish 1。
- D45事前計画：[凍結計画](D45_D43_SECOND_ROUND_1000_GAMES_PLAN_20261107.md)。親は現行D43 `bf8c26f9f8f6adff8a87c46ab04fcf5cd76a7915be241da11b99bb4ae2ac8b9f`、Adam2882。開始時点の現行推論重みは `models/fukaji-d43.nnue.npz`。
- 5波×20独立ジョブ×10局＝1000局。集約の実行ログで**1000局、確定994局、未決着6局**、SHAとRelease読戻し成功。学習対象の独立親・合法性の検査はCI成功の証拠。今回のAI側では全棋譜をローカル再実行していない。
- [1000局Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d45-d43-1000-selfplay-42326260509) SHA256 `be22c4e92eff0b684f06502dcdf9be52071ab60beab5a4a91edd8605f4d873c4`。

## 単一writer追加学習

- 学習後のNNUE研究snapshot SHA256：`941598f2a9ec926d5f67dadf03582ccef718a56a843070dc1f576cc7eeca44e8`。
- Adam step：**2882 → 3794**。新1000局中の確定994局＋旧bootstrap 665局＋D43前周trainの300局を利用、学習対象1959局、**58,334局面**。2epoch、loss 0.588080→0.575396→0.567178。loss低下だけでは棋力を判断しない。
- [学習済み完全stateとoptimizer Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d45-d43-trained-42326260509) SHA256 `d44349470b4068224ce1383f29d1afdf941c39931f6141c278927976da52c820`。trainジョブ成功・再ダウンロードSHA照合。

## 学習前後の新規200局比較

- 20独立auditジョブ全成功、**新規100序盤×先後交替200局**、24,451手の合法再生をCIで検証。
- **学習後155勝、学習前44勝、引分0、未決着1**。勝数差 **+111**、未決着1。事前基準（差12以上、未決着10以下）を満たし、判定 `after_better`。未決着を勝ちや引分に変換しない。
- 保守的ペア単位95%区間：学習後スコア約 **63.9〜91.6%**。この比較条件で強い根拠はあるが、他の時間制御・相手に対する一般保証ではない。
- [200局全棋譜・計画・結果Release](https://github.com/mtk-ctrl/shogi-fukaji/releases/tag/fukaji-d45-d43-before-after-200games-42326260509) SHA256 `6531747f10d015bec7817186f6fbed149aa6ac1822d94fbcce7858a7ceae211b`。finishジョブで再取得SHA照合。
- **結果として重みの採用基準は通過。ただし採用処理はD45ワークフローの範囲外。** 正式契約、713件回帰、実C++/USI、推論重み抽出・SHA、配布Release読戻しと、mainへの安全な原子的更新が終わるまで、**現行はD43/Adam2882のまま**。

## 次の工程・権限・安全

F00/F20の本人委任とD45事前計画に従い、次は**D45学習後重みのみを現行化するためのD44同等の検証付きワークフロー**。D44の実装 `scripts/d44_adopt_d43.py` と `.github/workflows/d44-adopt-d43-current.yml` を参考に、D45固有のSHA、1000局／200局結果、Adam3794を厳格に検査。事前に全契約テスト・713件回帰・実USI・独立配布試験、変更対象ファイルの固定、push直前のmainのCAS検証を通す。**実装・起動・採用は本結果記録時点では未実施**。前モデルD43と全state/optimizer/棋譜を保持し、失敗したらmainの正式モデルを変えない。USI表記・探索仕様は変更しない。人間の定跡を教師にしない・詰み探索を重視する[F00の新方針](rules/F00_深路の目的と権限.md)を維持。

## 証拠の境界

Actionsの各ジョブ結果、集約／学習／finishのログ、GitHub Releaseの公開SHA256とCI内読戻しを確認。全tar.gzをこの実行環境へ個別ダウンロードして再SHA計算したわけではない。APIの結果・実ジョブログ・GitHub保存証拠と、独立再実行を混同しない。

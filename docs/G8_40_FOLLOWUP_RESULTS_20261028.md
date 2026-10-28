# 深路 G8：40局面独立再分析の実測結果（2026-10-28）

研究専用。雲路正式版・G7凍結モデル・正式採用状態は不変更。

## 出所と実行

- [基盤PR #48](https://github.com/mtk-ctrl/shogi-fukaji/pull/48) squash main 372a98576f06194ac8ad5947046509781882bbf9。branch learning CI 40633836765 success（271テスト）、main CI 40633993332 / 40633993445 / 40633993304 success。
- 初回テスト1件失敗は python-shogi のundo後に0回の反復履歴キーが残るため。既存と同じ正の反復カウント比較に修正し、全271テスト成功。workflowのGitHub式閉じ括弧不足も修正済み。
- 所有27repoのqueued/in_progress/pending/waiting/requested全0を確認後、別commit 312a72edb128634c160fd1e3f603649aa1cb9dc0 で研究requestを明示有効化。
- **[実run #40634501714](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40634501714) success**：preflight、20/20並列分析各2局面、aggregate。
- G7モデル SHA256 417d18e7b64494c9febdf6d217afbe4e4979e68d04528f16db8b45c0bcde2a0e。
- 元60入力artifact12791352643 SHA256 0d11ac42493bb595b9a74769781a364703f8d4f521c66c8f9ba78aa17583672c。元60分析artifact12791512324 SHA256 cfb8278806940f7e845e6ae0475b4500777308a0d077537c6d2f4b8134d279d5。
- 固定40局面JSONL SHA256 5a801a692b0cdaf9d97a7622da0d5676b553a01ff933183fc0018b07f1a4d6bb。train親40・SFEN40、重複0。詰み終盤16・詰み中盤8・未決着終盤9・序中盤7。先手番21・後手番19。
- 1局面につき20秒再探索（最大深さ5）、30秒独立探索（最大深さ5）、元200ms手と元20秒長考手をそれぞれ指した後の相手5秒応手（最大深さ4）。Actionsで元60履歴の合法再生、40候補手/応手の合法性、全push/pop・SFEN/実効反復履歴の復元を検証。

## 実測（40局面）

| 指標 | 実測 |
|---|---:|
| 以前の20秒手と今回20秒再探索手の一致 | **37/40** |
| 以前の20秒手と今回30秒探索手の一致 | **32/40** |
| 今回20秒再探索と30秒探索の一致 | **35/40** |
| 30秒探索の実時間中央値 | **30,094.4435ms** |
| 元手・長考手の後の相手応手診断 | **80件** |

以前の20秒手は30秒でも32局面で維持、8局面で変化した。これは探索安定性の観測であり、手の正しさ・改善効果・教師ラベルの証明ではない。

## 成果物と監査

- GitHub入力1＋20分割＋集約1の22 ZIPを実取得し、GitHub artifact SHA256と**22/22一致**。集約artifact12792659765 ZIP SHA256 872b608b492b6037f33358a1e7cfd3d23294e98103f99f71f1b2139adc407ddd。集約40行 SHA256 925ba64d3e3b59ebcb5311776c188b1397d186acbe663a47d219c71654aac956。
- 別環境で20分割各2行、40件のID・親・元手・再探索手・安定性フラグ・統計を独立再計算一致。元60局面から40選定を独立再構築し、バイト単位で一致。
- **別環境でのpython-shogi全手合法再生は未実施**。Actions内では合法性・全undo確認済み。
- [Google DriveのG8再分析原本22件](https://drive.google.com/drive/folders/1RJd-J_CT7noTO61Cdb0ihhA0-JccJQpZ)へ全件保存し、22件を再取得してGitHub SHA256と**22/22一致**。ローカル統合監査ZIP単体はcontainer_session_unavailableでDriveアップロードできなかった（原本22件の保存とは区別）。
- request disabled commit ba2ec86fd1eb850ef0651d819b24592211de5156。[skip CI #40634994913](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40634994913) success（preflight成功、分析/集約skipped）。

## 次工程

新400局の詰み258局のうち**train206局のみ**をG8研究学習の新教師とする。先手勝102・後手勝104、train206局JSONL SHA256 d4160a0f354381a5cc0bbfce4fb2a98790fb779d1eb6d6c256919e574c611cc7。旧G7の557親群・割当と新400親群のtrain320/validation40/test40を保護し、新validation26/test26・未決着142は教師にもパラメータ調整にも使用しない。

G7の凍結学習器から事前固定の学習条件でG8研究候補を生成し、別seedでG7対G8の先後交替240局を行う。長考40局面の手は自動教師にしない。未決着は勝ち扱いせず、棋力向上と正式採用を混同しない。Goal未完、雲路不変更。

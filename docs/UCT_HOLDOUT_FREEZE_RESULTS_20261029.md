# UCT保持群12親の凍結・全240局合法再生監査結果（2026-10-29）

状態：**研究用保持群の凍結と合法性検証まで完了**。200msの新ABBA実測、棋力対局、学習、正式採用は未実施。雲路main、深路正式USI・G8モデル・既定値は変更していない。

## 出所・凍結条件

- 凍結元はG8/G7先後交替240局、run [40637486947](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40637486947)。GitHub artifact ZIP（82,734 bytes）を取得し、SHA256 `a3a8bee0a63478b296c7362cc33adb33c31d37407ae3b28527b0e1c9688c3b14` を再計算一致。内部 `all_paired_games.jsonl` 240行SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf` 一致。
- 原本240局の内訳は詰み147局・240手未決着93局。未決着に勝ち・引分教師を付けない。
- 以前の開発群はpair0〜11。保持群は**別pair12〜23**、12/50/110手目各4、先手G8側の各対局から一つだけ。開発群と保持群の対局ID・SFEN・親IDの重複なし。各親のSFEN、seed、対局行番号、原本SHAは [凍結JSON](./UCT_HOLDOUT_12_PARENTS_20261029.json) に保存。
- 元開発群12 SFENについて独立の簡易盤面復元が12/12一致。保持群のSFENも同じ方法で生成。ただし簡易復元は王手・打ち歩詰などの完全合法性証明ではない。

## GitHub CIでの完全合法性検査

- 入口 [PR#85](https://github.com/mtk-ctrl/shogi-fukaji/pull/85)、branch head `7e142d2019ea095f5e20f95fc191d4f833aefd16`。初回CI [40779443767](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40779443767) は240局再生に成功したが、receiptの `legality_replayed:false` がメタデータ検査器の既定値のまま残る表示上の不整合を発見。実再生を終えた検証器でのみ `true` とする修正を追加。
- 修正後branch [CI40779591540](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40779591540) success。元原本SHA照合、**240局全手合法再生、開発群12 SFEN一致、保持群12 SFEN・履歴・合法手あり・重複なし**を検証。独立取得したreceipt ZIP SHA256 `3420eee4696ed9130823d6a75457b0db8db5c9c54b068251db72f1f3021b5e37`。receiptに `legality_replayed:true`、`source_games_legally_replayed:240`、`holdout_prefixes_legally_replayed:12`、`measurement_started:false` を確認。
- 期待headでsquash研究main `da3d09e6e3eaedc87a273f93643c32458fd2677a` に統合。main [合法再生CI40779843631](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40779843631) success、receipt ZIP SHA256 `1bfbc5b931752492b1315eb214756d77fff8f69a825d875a955f95e962db397f` を独立取得再計算一致。main [軽量回帰CI40779843740](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40779843740) success。mainの凍結JSON・検証コード・workflow・計画文書を読み戻し。
- 今回のモデルSHA `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492` は**後続実測用の参照値のみ**。今回G8モデル実バイトを取得・検証していない。独立ローカル環境にpython-shogiがないため、240局全手の**独立ローカル再生は未実施**（Actions上では全240局を再生）。

## 次工程と限界

- 新たな200ms探索は0回、48観測ABBAも未起動。27repoの5状態/同目的重複のrunner調査は**今回行っていない**。新しい重いrunの前に必ず再調査する。
- 保持群を使って予約係数を調整しない。既定OFFの保持群ABBA入口を別PRで作り、branch CI→squash研究main→main CI→27repo5状態/重複調査→別明示request→1runner→G8モデル実SHA・原本SHA・全合法性/48行/統計/成果物SHA独立監査→request disabled/skipを行う。別CI runを独立実行環境と断定しない。
- **200ms硬上限、探索木解放予約の汎化、棋力向上、正式採用は未証明**。NNUE PR#62とDrive原本ZIP保存の安全拒否は迂回せず保留。原本artifact期限は2026-11-27。

漫画素材（創作案）：「同じ十二局面だけで速くなったと言ってはいけない。別の十二局面を、まだ触らぬ試験紙として封じた。」

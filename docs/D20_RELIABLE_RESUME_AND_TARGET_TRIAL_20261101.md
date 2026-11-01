# D20：保存・再開修復と同一経験MC/TD比較

2026-11-01。[D20 Goal](https://www.addness.com/goals/b325f427-7af8-4be6-be37-5808d6ef8f56)が本人限定状態の現行索引。正式版の採用変更は行わない。

## 障害と修復

- 公開repoへのモデル・棋譜送信は公開許可不足として自動審査に拒否された。公開送信は中止し、本人のみのDriveへ完全状態を保存。コード保存は成功している。公開許可を推定して再送しない。
- Addnessの親本文は約23万字、394300 bytes。権限はあり残容量も約127KBだが、短文の追加がINVALID_ARGUMENT、既存文字列の編集は成功。新子Goalには追加できた。親本文固有の書込み経路の不調で、サーバー内部の厳密な原因は未確認。
- 全文をDriveへ退避し再取得SHA256一致。AIメモリUIのみ一時除去、force=falseで現行要約へ置換してUIを復元。メモリ保持、通常追記成功、本文読戻しを確認。過去記録を消して対処していない。
- Actionsには入力空欄で古い公開第5周へ戻る経路があった。再開元を必須の明示選択に変更。過去公開状態は「歴史的な第5周」と表示し、最新と呼ばない。
- 公開Actionsでのモデル・棋譜artifact保存にはallow_public_state_artifactを明示要求。未許可なら生成前に停止。本人限定データはローカルで生成しDriveへ保存する。

## 第11周からの再開

Goalに記録した本人限定indexとarchiveを取得し、次を実行する。

```bash
python -m fukaji.online_resume restore --index /path/resume-index.json --archive /path/state.tar.gz --root build/research-state
```

SHA256、全モデルとoptimizer、経験の整合性・train/monitor分離に加え、index指定の最新モデル・周回数・optimizer stepを照合する。不一致時は復元先を公開せず、既存状態も上書きしない。

今回、既存作業フォルダが失われた状態から第11周ac2d9c51、optimizer1288、13models、train709、monitor156を再取得・復元・照合できた。

## 続く一要因実験

第11周の同じ親重み・optimizerからMC（実勝敗100%）とTD（実勝敗75%＋凍結親TD25%、lambda=.8）を作る。直近3周の決着12局＋bootstrap trainから固定seedで12局、合計24局を両側で共有。位置抽出seed、抽出位置、2epoch、lr=.001、探索を固定。monitorを学習に使わない。

```bash
OPENBLAS_NUM_THREADS=1 python -m fukaji.online_target_trial --root build/d20-verified-state
```

seed=202611012020、8新序盤の先後16局、各手100ms指定、最大256手。各組の棋譜を逐次保存。未決着を引分にしない。研究pointerを変更せず、凍結候補同士を比較する。これは旧D18の親・経験・探索とは別条件だが、小標本なので優位の確定ではない。結果は実行後に追記する。

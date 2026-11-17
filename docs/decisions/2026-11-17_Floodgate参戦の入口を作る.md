# 深路のFloodgate参戦の入口を作る（2026-11-17）

## 指示と状況
- 2026-11-15 本人指示: 深路もFloodgateに出す（雲路と同じ設定、ランナー1台を空ける）。
- 2026-11-17 本人: 「深路のリポジトリにもFLOODGATE_TRIPを登録しました」。リポジトリsecretは雲路と別管理のため、深路側に別途登録済み。

## 作ったもの
- `tools/floodgate/`（雲路shogi-aiのclient・protocol・timeplan・mock_server・chainを複製。名前の既定値のみFUKAJIに変更）。`tests/test_floodgate_client.py`・`test_floodgate_chain.py`。
- `.github/workflows/floodgate.yml`: 正式ハイブリッド（雲路の固定コミット＋深路の正式NNUE、`formal/model.json` hybrid節）をビルドし、Floodgate接続部品の `--engine` に渡す。USIオプションは正式の推奨どおり PositionKnowledge=false・AdaptiveLongThink=false・USI_Ponder=false、Floodgateでは無作為初手を切る（OpeningRandomNonLance=false）。読み時間はFloodgate専用の `--time-plan`（雲路と同じ）。評価値は `--send-eval` で公開棋譜に残す。
- `tools/floodgate/continuous.json`: 常時参加の設定。**既定は enabled=false**。

## 守ったこと
- 雲路の局面知識（163件・追加分）は使わない（深路は雲路の手作業評価・研究手・局面知識を教師にしない）。
- 定期起動(schedule)は付けない（F30）。連鎖は雲路と同じ（成功なら次、失敗は最大3回、取り消しでは連鎖しない）。
- 相手の評価値・読み筋・指し手は学習に使わない（R50相当）。
- 実参戦(live)は本人の指示があるまで行わない。

## 検証
- 単体試験（接続部品14件、連鎖7件）は手元で通過。
- Actionsでの probe と rehearsal（模擬サーバ先後各1局）の結果は、この文書の追記に残す。

## 次の一手
- 本人が実参戦を指示したら、まず1局（mode=live, games=1, confirm_live=LIVE）。続けて常時参加にするときは continuous.json を enabled=true にして continuous=yes で起動。
- 雲路の常時参加と同じ組み合わせ時刻になる。同じ相手に当たることがある（気になるなら時刻をずらす設定を追加する）。

## 追記: Actionsでの検証結果（2026-11-17）
run 44004822765（mode=rehearsal、commit e4567bc）が成功。
- 正式ハイブリッドのビルド（雲路 199dbb0＋E16）と、USI接続・時計確認（bestmove 7g7f）が通った。
- Floodgateサーバ（wdoor.c.u-tokyo.ac.jp:4081）へのTCP到達を確認（ログインはしていない）。
- 模擬サーバで先後各1局に勝ち、評価値コメントが各9件届いた。
- 実参戦(live)・連鎖は未実施（本人の指示待ち）。ランナーが満員（G1データ生成など）だったため、起動から実行開始まで約15分待った。実参戦のときは、雲路側と同じく1台を空けておく（F30）。

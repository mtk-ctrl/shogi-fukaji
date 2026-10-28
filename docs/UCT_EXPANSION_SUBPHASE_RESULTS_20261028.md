# 深路G8：UCT展開工程5内訳の実測結果（2026-10-28）

**結論：** 凍結G8モデル・原本240局の合法再生後、前回時間超過が集中した親6/11を各2回、合計119反復測定した。展開工程758.682msのうち、`list(board.legal_moves)` が747.509ms（**98.53%**）を占めた。これは**計測した展開工程内**の割合であり、探索全体の壁時計時間や改善余地の割合ではない。原因は合法手リスト生成の内部に絞られたが、列挙自体・`__len__` 事前走査・合法性判定の内訳は未分離。

## 検証済み実行経路
- 計測コード：[研究PR#66](https://github.com/mtk-ctrl/shogi-fukaji/pull/66)（既定OFFの `iteration_trace` のみ）。branch CI [40692844422](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40692844422) **331テスト成功**、squash研究main `f16543453b6e1d294625eaf7e51bf5831518681a`、main CI [40693218199](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40693218199) success。
- 27repoの直近最大100runと、100件超の8repoについてqueued/in_progress/pending/waiting/requestedを個別照合し、起動前のactive0・同目的重複なし。1runnerで別の明示request commit `5a03196ae55b6b81fd6090f60719dde9352f82e6` を起動。他run停止なし。
- [実測run40693427493](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40693427493) completed/success。原本run40637486947の240局JSONL SHA256 `60eb9ab4dadc3776e43ac49cf27bd04137477c339850fdaa7db2265248a8d3bf`、G8モデルSHA256 `18b39f647caa458389a1900b875337adadac74ae37e27343e7f0094ed663c492` 固定。実行側で全240局の合法再生、各局面のSFEN・履歴復元・選択手合法性を検査。未決着を勝ち・引分・教師に流用しない。

## 4系列・計119反復
| 対象 | 反復数 | 壁時計時間 | 最長1反復 | 根合法手 | 持駒数 | 合法手リスト生成 |
|---|---:|---:|---:|---:|---:|---:|
| 親6・1（50手目） | 29 | 193.622ms | 7.372ms | 94 | 9 | 186.746ms |
| 親6・2 | 30 | 194.815ms | 7.895ms | 94 | 9 | 187.908ms |
| 親11・1（110手目） | 30 | 191.457ms | 6.952ms | 38 | 10 | 184.640ms |
| 親11・2 | 30 | 194.620ms | 7.175ms | 38 | 10 | 188.214ms |

| 展開工程の内訳 | 計測合計 | 展開758.682msに対する割合 |
|---|---:|---:|
| 合法手リスト生成 `list(board.legal_moves)` | **747.509ms** | **98.527%** |
| 子ノード生成・初手選択 | 8.292ms | 1.093% |
| 詰み判定 | 1.208ms | 0.159% |
| 着手・経路更新 | 1.067ms | 0.141% |
| 千日手判定 | 0.325ms | 0.043% |
| 5内訳以外の計測オーバーヘッド等 | 0.282ms | 0.037% |

5内訳はいずれも119回記録（該当未実行時は0にする仕様）。選択1.898ms・葉評価10.148ms・逆伝播0.187ms・undo1.990msは展開外の計測。展開を含む5大工程合計は772.905msで、探索外側の壁時計時間と同一ではない。計測有効の200ms超過は0/4だが、前回の非計測ABBAで親6/11の4/4超過が解消した証拠ではない。

## 独立成果物監査と保存
- artifact ID `12818321495`、ZIP 14,501バイト、GitHub公表SHA256 `c276996b78a3ded5481465a2fffc77c1dcea78555f80ea93bac1c763c5eec1de`。別取得のZIP実バイトでSHA256一致。
- ZIP内部 `phase_iterations.jsonl` 4行のSHA256 `885c557ae1d27b3dbb74c36ec5f90631d5b7a096660ee4539da75e069b157d76` が `summary.json` と一致。全119反復について内訳非負・内訳小計が展開合計を超えないこと、親6/11各2行、凍結SHA・合法手数・反復数・合計を独立再計算して一致。
- [Google Driveの原本ZIP](https://drive.google.com/file/d/1gddRs_6WhXdVR5fQ-D1v4RSadv0L5RA3/view)に保存、再取得ZIPのサイズ14,501バイト・SHA256一致。[前段の原本ZIP](https://drive.google.com/file/d/1KaueVaF-NoEyNP8l2mROxtgPn8DZmPuO/view)も今回長期保存し再取得SHA256 `49c9cda4f975bbed7b41686721b4b921f0f2dc8818dadf02a047584a2af18843` 一致。
- requestをcommit `8f9aa9d5b3ccfad6a176f79ecfe4e1d172b71221` で `enabled:false` に復帰。[skip run40693584474](https://github.com/mtk-ctrl/shogi-fukaji/actions/runs/40693584474) completed/success。事前検査は成功、モデル取得・240局再生・計測・artifact保存の重い工程はすべてskippedと直接確認。

## 判断・次工程
1. **律速の候補は `list(board.legal_moves)` に絞られた。** python-shogiのリスト化における `__len__` の事前走査、列挙、王手・打ち歩詰・持駒の合法性検査を、挙動を変えず分離測定する。
2. 合法手の集合と**順序**、詰み・王手・反復・持駒・打ち歩詰、盤面undoの完全一致を検査してから、`list(...)` 対 `[m for m in ...]` の研究専用ON/OFFを固定局面で測る。既存G7での高速化経験をG8/UCTへ無検証で転用しない。
3. 時間遵守は計測なしの同一12局面ABBAで再検証し、棋力の証明は必要なら先後交替240局で別途実施。**厳密200ms保証・棋力向上・正式採用はいずれも未達**。
4. NNUE学習PR#62の安全拒否された書込みは再試行・迂回せず、未統合・実学習なし。雲路main、深路既定USI/モデル、正式採用は不変更。

## 漫画素材（創作案、本人の実発言ではない）
「展開98%」という大きな箱を開くと、詰み判定でも評価計算でもなく、合法手をリストにする処理へ時間が集中していた。しかし、まだ時計は原因の入口を示しただけ。次は合法手を一度だけ作れているのかを確かめる。
